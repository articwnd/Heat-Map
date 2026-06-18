# correlation_heatmap.py -- see README.md for full documentation and usage.

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
from datetime import date

# ── Third-party: data & numerics ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf

# ── Third-party: statistics ───────────────────────────────────────────────────
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# ── Third-party: visualization ────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit here, nowhere else)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Tickers: diversified mix to showcase cross-sector correlation patterns
    "tickers": [
        "SPY",   # S&P 500 (market benchmark)
        "QQQ",   # Nasdaq-100 (growth/tech)
        "IWM",   # Russell 2000 (small-cap)
        "XLE",   # Energy sector
        "XLF",   # Financials sector
        "XLV",   # Healthcare sector
        "GLD",   # Gold ETF (safe haven)
        "TLT",   # 20-yr Treasury (rate-sensitive / diversifier)
        "AAPL",  # Apple (mega-cap tech)
        "MSFT",  # Microsoft (mega-cap tech)
        "JPM",   # JP Morgan (large-cap financials)
        "XOM",   # Exxon Mobil (large-cap energy)
    ],
    "start_date": "2022-01-01",   # Captures rate-hike regime + recovery
    "end_date":   str(date.today()),
    "min_obs":    126,             # Minimum trading days required per ticker
    "output_html": "correlation_heatmap.html",
    "output_png":  "correlation_heatmap_static.png",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices via yfinance.

    In production, replace this function body with:
        - openbb: obb.equity.price.historical(symbol=..., provider="fmp")
        - polygon.io: polygon.RESTClient(...).get_aggs(...)
        - Bloomberg: blpapi (requires terminal license)

    Parameters
    ----------
    tickers : list of ticker strings
    start   : 'YYYY-MM-DD'
    end     : 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame  [date x ticker], adjusted close prices, NaN where unavailable
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # Adjusts for splits and dividends
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers are passed
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns that are entirely empty (failed downloads)
    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.to_datetime(prices.index)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA QUALITY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(prices: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """
    Print a data-quality report and drop tickers below the minimum observation
    threshold. In production this would write to a structured log.

    Parameters
    ----------
    prices  : price DataFrame
    min_obs : minimum non-NaN observations required to retain a ticker

    Returns
    -------
    pd.DataFrame  Cleaned price DataFrame
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY DIAGNOSTICS")
    print("=" * 60)

    report = pd.DataFrame({
        "obs":      prices.notna().sum(),
        "missing":  prices.isna().sum(),
        "pct_miss": (prices.isna().mean() * 100).round(2),
        "start":    prices.apply(lambda s: s.first_valid_index()),
        "end":      prices.apply(lambda s: s.last_valid_index()),
    })

    print(report.to_string())
    print()

    # Drop tickers below threshold
    keep = report[report["obs"] >= min_obs].index.tolist()
    dropped = [t for t in prices.columns if t not in keep]
    if dropped:
        print(f"[WARN] Dropping tickers below {min_obs}-obs threshold: {dropped}")

    return prices[keep]


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETURN COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1}).

    Log returns are preferred over simple returns in quant work because:
      - They are time-additive (multi-period compounding is a sum, not a product)
      - They are more symmetric and closer to normally distributed
      - They prevent negative compounded values

    Parameters
    ----------
    prices : pd.DataFrame of adjusted close prices

    Returns
    -------
    pd.DataFrame of log returns, first row dropped (NaN from diff)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_returns


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRELATION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlation matrices.

    Pearson: measures linear co-movement, sensitive to outliers.
    Spearman: rank-based, captures monotonic relationships, more robust
              to fat tails, which are common in financial return distributions.

    In practice, quants cross-check both: large Pearson-Spearman gaps indicate
    non-linear dependence or outlier sensitivity in the Pearson estimate.

    Parameters
    ----------
    returns : pd.DataFrame of log returns

    Returns
    -------
    (pearson_corr, spearman_corr)  each a symmetric pd.DataFrame
    """
    # Both use pandas .corr(), which:
    #   - returns a correctly-shaped NxN matrix at any N (including N=2),
    #     unlike scipy.stats.spearmanr which returns a bare scalar at N=2
    #   - uses pairwise-complete observations for NaN handling, matching
    #     Pearson's convention, unlike scipy's default nan_policy="propagate"
    #     which NaNs out an entire row/column if a single value is missing
    pearson = returns.corr(method="pearson")
    spearman = returns.corr(method="spearman")

    return pearson, spearman


# ─────────────────────────────────────────────────────────────────────────────
# 5. HIERARCHICAL CLUSTERING REORDER
# ─────────────────────────────────────────────────────────────────────────────

def hca_reorder(corr_matrix: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Reorder assets using Hierarchical Clustering Analysis (HCA) so that
    similar assets appear adjacent in the heatmap, revealing block structure.

    Distance metric: d(i,j) = sqrt(0.5 * (1 - rho_{ij}))
        - Bounded in [0, 1]
        - d=0 when rho=1 (perfect positive correlation)
        - d=1 when rho=-1 (perfect negative correlation)
        - d=sqrt(0.5) when rho=0 (uncorrelated)
        - Satisfies triangle inequality (it is a proper metric)

    Linkage: Ward minimizes within-cluster variance at each merge step.
    This is the industry standard choice for financial asset clustering.

    Parameters
    ----------
    corr_matrix : pd.DataFrame, symmetric correlation matrix

    Returns
    -------
    (ordered_tickers, linkage_matrix)
    """
    # Work on a writable numpy array to avoid read-only DataFrame view errors
    dist_arr = np.sqrt(0.5 * (1.0 - corr_matrix.values)).copy()
    np.fill_diagonal(dist_arr, 0.0)

    # Condensed form required by scipy linkage
    dist_condensed = squareform(dist_arr, checks=False)
    linkage_matrix = hierarchy.linkage(dist_condensed, method="ward")

    order = hierarchy.leaves_list(linkage_matrix)
    ordered_tickers = [corr_matrix.columns[i] for i in order]

    return ordered_tickers, linkage_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANNUALIZED VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def annualized_vol(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualized volatility = daily_std * sqrt(252).

    Provides per-asset risk context displayed alongside the heatmap.

    Parameters
    ----------
    returns      : pd.DataFrame of log returns
    trading_days : convention (252 for US equities, 365 for crypto)

    Returns
    -------
    pd.Series  annualized vol per ticker, as a decimal (e.g. 0.18 = 18%)
    """
    return returns.std() * np.sqrt(trading_days)


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTERACTIVE PLOTLY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def build_interactive_heatmap(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    ordered_tickers: list[str],
    ann_vol: pd.Series,
    date_range: tuple[str, str],
    output_path: str,
) -> None:
    """
    Build and save an interactive Plotly heatmap with:
      - HCA-ordered Pearson correlation as the primary heatmap
      - Spearman correlation in hover tooltip for comparison
      - Annualized volatility bar chart as a sidebar
      - Diverging RdBu_r colorscale centered at 0

    Output is a self-contained HTML file (no server required).

    Parameters
    ----------
    pearson         : Pearson correlation DataFrame
    spearman        : Spearman correlation DataFrame
    ordered_tickers : HCA-reordered ticker list
    ann_vol         : annualized volatility Series
    date_range      : (start_date_str, end_date_str)
    output_path     : path to save HTML file
    """
    p = pearson.loc[ordered_tickers, ordered_tickers]
    s = spearman.loc[ordered_tickers, ordered_tickers]
    n = len(ordered_tickers)

    # --- Build hover text matrix ----------------------------------------
    hover_text = []
    for row_ticker in ordered_tickers:
        row = []
        for col_ticker in ordered_tickers:
            pval = p.loc[row_ticker, col_ticker]
            sval = s.loc[row_ticker, col_ticker]
            cell = (
                f"<b>{row_ticker} vs {col_ticker}</b><br>"
                f"Pearson:  {pval:+.4f}<br>"
                f"Spearman: {sval:+.4f}<br>"
                f"<i>Vol({row_ticker}): {ann_vol[row_ticker]:.1%}</i><br>"
                f"<i>Vol({col_ticker}): {ann_vol[col_ticker]:.1%}</i>"
            )
            row.append(cell)
        hover_text.append(row)

    # --- Build annotation text (show numeric labels on cells) -----------
    annot_text = [[f"{p.loc[r, c]:.2f}" for c in ordered_tickers]
                  for r in ordered_tickers]

    # --- Build font color matrix (black on mid values, white on extremes) --
    font_colors = []
    for row_ticker in ordered_tickers:
        row = []
        for col_ticker in ordered_tickers:
            val = abs(p.loc[row_ticker, col_ticker])
            row.append("white" if val > 0.65 else "black")
        font_colors.append(row)

    # --- Subplots: heatmap (left) + vol bar (right) ----------------------
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.82, 0.18],
        subplot_titles=["Pearson Correlation (HCA-Ordered)", "Annualized Vol"],
        horizontal_spacing=0.06,
    )

    # Primary heatmap
    fig.add_trace(
        go.Heatmap(
            z=p.values,
            x=ordered_tickers,
            y=ordered_tickers,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale="RdBu_r",       # Red=positive, Blue=negative (quant standard)
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Pearson r",
                thickness=15,
                len=0.8,
                x=0.80,
            ),
            # Annotate cells with numeric labels
            texttemplate="%{z:.2f}",
            textfont=dict(size=10),
        ),
        row=1, col=1,
    )

    # Volatility bar chart (horizontal)
    vol_ordered = ann_vol[ordered_tickers]
    bar_colors = [
        f"rgba({int(255*(v/vol_ordered.max()))}, 80, 80, 0.85)"
        for v in vol_ordered.values
    ]

    fig.add_trace(
        go.Bar(
            x=vol_ordered.values,
            y=ordered_tickers,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>Ann. Vol: %{x:.1%}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # --- Layout styling --------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Stock Correlation Heatmap</b>  |  "
                f"{date_range[0]} to {date_range[1]}<br>"
                f"<sup>Daily log returns  |  HCA reorder: Ward linkage, "
                f"d = sqrt(0.5*(1-rho))  |  Hover for Spearman r</sup>"
            ),
            x=0.5,
            font=dict(size=14),
        ),
        paper_bgcolor="#0d1117",    # Dark background: industry-standard dark theme
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="monospace"),
        height=680,
        width=1050,
        margin=dict(t=100, b=60, l=70, r=20),
    )

    # Axis styling
    axis_style = dict(
        showgrid=False,
        linecolor="#30363d",
        tickfont=dict(size=11, color="#e6edf3"),
    )
    fig.update_xaxes(axis_style, tickangle=-40, row=1, col=1)
    fig.update_yaxes(axis_style, row=1, col=1)
    fig.update_xaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Save
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[OK] Interactive heatmap saved -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. STATIC SEABORN CLUSTERMAP (for reports / PDFs)
# ─────────────────────────────────────────────────────────────────────────────

def build_static_clustermap(
    pearson: pd.DataFrame,
    ann_vol: pd.Series,
    output_path: str,
) -> None:
    """
    Build a publication-quality static clustermap using seaborn.clustermap().

    seaborn.clustermap() performs its own HCA and renders a dendrogram
    alongside the heatmap, making the cluster structure explicit.

    Uses the same correlation-distance metric as the interactive version
    for consistency between outputs.

    Parameters
    ----------
    pearson     : Pearson correlation DataFrame
    ann_vol     : annualized volatility Series (used for row color bar)
    output_path : path to save PNG file
    """
    # Build a color bar for annualized vol (low=blue, high=red)
    norm = mcolors.Normalize(vmin=ann_vol.min(), vmax=ann_vol.max())
    cmap_vol = plt.cm.get_cmap("RdYlBu_r")
    row_colors = pd.Series(
        {t: mcolors.to_hex(cmap_vol(norm(ann_vol[t]))) for t in pearson.index},
        name="Ann. Vol",
    )

    # Precompute the distance matrix for the dendrogram
    dist_arr = np.sqrt(0.5 * (1.0 - pearson.values)).copy()
    np.fill_diagonal(dist_arr, 0.0)

    sns.set_theme(style="dark", font_scale=0.85)

    g = sns.clustermap(
        pearson,
        method="ward",          # Consistent with the interactive version
        metric="precomputed",   # Use our correlation-distance, not Euclidean
        row_linkage=hierarchy.linkage(squareform(dist_arr, checks=False), method="ward"),
        col_linkage=hierarchy.linkage(squareform(dist_arr, checks=False), method="ward"),
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor="#333333",
        row_colors=row_colors,
        figsize=(12, 10),
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )

    g.fig.suptitle(
        "Stock Correlation Clustermap  (Ward / Correlation Distance)",
        y=1.01,
        fontsize=13,
        fontweight="bold",
    )

    # Add a small legend for the vol color bar
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cbar_vol = g.fig.colorbar(sm, ax=g.ax_heatmap, location="top", fraction=0.03, pad=0.02)
    cbar_vol.set_label("Annualized Volatility", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1c1c1c")
    plt.close()
    print(f"[OK] Static clustermap saved  -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY STATISTICS PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(pearson: pd.DataFrame, spearman: pd.DataFrame, ann_vol: pd.Series) -> None:
    """
    Print a concise quantitative summary of the correlation structure.
    Flags pairs with large Pearson-Spearman divergence (>0.10) as these
    may indicate non-linear or outlier-driven correlation.
    """
    print("\n" + "=" * 60)
    print("CORRELATION SUMMARY")
    print("=" * 60)

    n = len(pearson.columns)
    tickers = pearson.columns.tolist()

    # Extract upper triangle pairs (excluding diagonal)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = tickers[i], tickers[j]
            p = pearson.loc[t1, t2]
            s = spearman.loc[t1, t2]
            pairs.append((t1, t2, p, s, abs(p - s)))

    pairs_df = pd.DataFrame(pairs, columns=["Asset_A", "Asset_B", "Pearson", "Spearman", "Divergence"])
    pairs_df = pairs_df.sort_values("Pearson", ascending=False)

    print("\nTop-5 Highest Correlated Pairs (Pearson):")
    print(pairs_df.head(5).to_string(index=False))

    print("\nTop-5 Lowest (Most Diversifying) Pairs (Pearson):")
    print(pairs_df.tail(5).to_string(index=False))

    divergent = pairs_df[pairs_df["Divergence"] > 0.10]
    if not divergent.empty:
        print(f"\n[FLAG] {len(divergent)} pair(s) with Pearson-Spearman divergence > 0.10:")
        print(divergent.to_string(index=False))
        print("  -> Investigate for non-linearity or return distribution fat tails.")

    print("\nAnnualized Volatility:")
    print(ann_vol.sort_values(ascending=False).map("{:.1%}".format).to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    End-to-end pipeline:
      fetch -> diagnose -> log-returns -> correlations ->
      HCA reorder -> interactive plot -> static plot -> summary
    """
    cfg = CONFIG

    print(f"\nRunning correlation heatmap pipeline")
    print(f"Tickers : {cfg['tickers']}")
    print(f"Period  : {cfg['start_date']}  to  {cfg['end_date']}")

    # Step 1: Fetch prices
    prices = fetch_prices(cfg["tickers"], cfg["start_date"], cfg["end_date"])

    # Step 2: Diagnostics and cleaning
    prices = run_diagnostics(prices, cfg["min_obs"])
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 valid tickers to compute correlations.")

    # Step 3: Log returns
    returns = compute_log_returns(prices)

    # Step 4: Correlation matrices
    pearson, spearman = compute_correlations(returns)

    # Step 5: HCA reorder
    ordered_tickers, linkage_matrix = hca_reorder(pearson)
    print(f"\nHCA cluster order: {ordered_tickers}")

    # Step 6: Annualized vol
    ann_vol = annualized_vol(returns)

    # Step 7: Interactive Plotly heatmap
    build_interactive_heatmap(
        pearson=pearson,
        spearman=spearman,
        ordered_tickers=ordered_tickers,
        ann_vol=ann_vol,
        date_range=(cfg["start_date"], cfg["end_date"]),
        output_path=cfg["output_html"],
    )

    # Step 8: Static Seaborn clustermap
    build_static_clustermap(
        pearson=pearson,
        ann_vol=ann_vol,
        output_path=cfg["output_png"],
    )

    # Step 9: Summary statistics
    print_summary(pearson, spearman, ann_vol)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
# Heat_Map.py -- see README.md for full documentation and usage.

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
from scipy.stats import spearmanr

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
        "SPY", # S&P 500 (market benchmark)
        "QQQ", # Nasdaq-100 (growth/tech)
        "IWM", # Russell 2000 (small-cap)
        "XLE", # Energy sector
        "XLF", # Finance sector
        "XLV", # Healthcare sector
        "GLD", # Gold etf (safe haven)
        "TLT", # 20-year treasury (rate-sensitive / diversifier)
        "APPL", # Apple (mega-cap tech)
        "MSFT", # Microsoft (mega-cap tech)
        "JPM", # JP Morgan (large-cap financials)
        "XOM", # Exxon Mobil (large-cap energy)
        "NVDA", # Nvidia (Semi-conductors)
        "COST", # Costco (consumer defensive)
    ],
    "start_date": "2022-01-01", # captures rate-hike regime + recovery
    "end_date": str(date.today()),
    "min_obs": 126, #minimum trading days required per ticker
    "output_html": "correlation_heatmap.html",
    "output_png": "correlation_heatmap.png",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    '''
    Download adjusted closing prices via yfinance

    In production, replace this function body wot:
        - openbb: obb.equity.historical(symbol=..., provider="fmp")
        - polygon.io: polygon.RESTClient(...).get_aggs(...)
        - Bloomberg: blpapi (requires terminal license)

    Parameters
    ----------
    tickers  : list of ticker strings
    start    : 'YYYY-MM-DD'
    end      : 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame [date x ticker], adjusted close prices, NaN where unavailable
    '''
    
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers are passed
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(colums={"Close": tickers[0]})

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
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSTICS")
    print("="*60)

    report = pd.DataFrame({
        "obs": prices.notna().sum(),
        "missing": prices.isna().sum(),
        "pct_missing": (prices.isna().mean() * 100).round(2),
        "start": prices.apply(lambda s: s.first_valid_index()),
        "end": prices.apply(lambda s: s.last_valid_index()),
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
    Compute daily log returns: r_t = ln(P_t / P_{t-1})

    Log returns are preferred over simple returns in quant because:
        - They are time-additive (multi-period compoundign is a sum, not a product)
        - They are more symmetric and closer to normally distribution
        - They prevent negative compounded values

    Parameters
    ----------
    prices : DataFrame of adjusted close prices

    Returns
    -------
    DataFrame of log returns
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_returns

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRELATION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlation matrices.

    Pearson: measures linear co-movement, sensitive to outliers and non-linear relationships
    spearman: rank-based, captures monotonic relationships, more robust to fat tails, which are 
              more common in financial return distributions.

    Parameters
    ----------
    returns : DataFrame of log returns

    Returns
    -------
    (pearson_corr, spearman_corr) each a symmetric pd.DataFrame)
    """
    # both use pandas .corr(), which:
    #    - returns a correctly-shaped NxN matric at any N (including N=2)
    #      unlike scipy.stats.spearmanr which returns a bare scalar at N=2
    #    - uses pairwise-complete observations for NaN handling, matching
    #      Pearson's convention, unlike scipy's default nan_policy="propagate"
    #      which NaNs out an entire row/column if a single value is missing

    pearson = returns.corr(method="pearson")
    spearman = returns.corr(method="spearman")

    return pearson, spearman

# ─────────────────────────────────────────────────────────────────────────────
# 5. HIERARCHICAL CLUSTERING REORDER
# ─────────────────────────────────────────────────────────────────────────────
def hca_reorder(corr_matrix: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Reorder assets using Hierarchical Clustering Analysis (HCA) so that
    similar assets appear adjacent in the heatmap, revealing block structures

    Distance metric: d(i,j) = sqrt(0.5 * (1 - rho_{ij}))
        - Bounded in [0,1]
        - d=0 when rho=1 (perfect positive correlation)
        - d=1 when rho=-1 (perfect negative correlation)
        - d=sqrt(0.5) when rho=0 (uncorrelated)
        - Satisfies triagnle inequality (it is a proper metric)

    Linkage: ward minimizes within-cluster variance at each merge step.
    This is the industry standard choice for financial asset clustering.    


    Parameters
    ----------
    corr_matrix : DataFrame of correlations

    Returns
    -------
    (ordered_tickers, linkage_matrix)
    """
    # work on a writable numpy array to avoid read-only DataFrame view errors
    dist_arr = np.sqrt(0.5 * (1 - corr_matrix.values)).copy()
    np.fill_diagonal(dist_arr, 0.0)

    # condensed form retuired by scipy linkage
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
    Annualized volatility = daily_std * sqrt(252)
    
    Provides per-asset risk context displayed alongside the map

    Parameters
    ----------
    returns: pd.DataFrame of log returns
    trading-days: convention (252 for US equities, 365 for crypto)

    Returns
    -------
    pd.Series annualized vol per ticker, as a decimal (e.g. 0.18 = 18%)
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
        data_range: tuple[str, str],
        output_path: str,
) -> None:
    """
    Build and save an interactive Plotly heatmap with:
        - HCA-ordered Peason correlation as the primary heatmap
        - Spearman correlation in hover tooltip for comparison
        - Annualized volatility bar chart as a sidebar
        - Diverging RdBu_r color scale centered at 0

    Output is a self-contained HTML file (no server required)

    Parameters
    ----------
    pearson : Pearson correlation DataFrame
    spearman: Spearman correlation DataFrame
    ordered_tickers: HCA-reordered ticker list
    ann_vol: annualized volatility Series
    data_range: (start_date, end_date) 
    output_path: file path for output HTML
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
                f"<b>{row_ticker} & {col_ticker}</b><br>"
                f"Pearson: {pval:.4f}<br>"
                f"Spearman: {sval:.4f}"
                f"<i>Vol({row_ticker}): {ann_vol[row_ticker]:.1%}</i><br>"
                f"<i>Vol({col_ticker}): {ann_vol[col_ticker]:.1%}</i><br>"
            )
            row.append(cell)
        hover_text.append(row)

    # --- Build annotation text (show numeric labels on cells) -----------
    annot_text = [[f"{p.loc[r,c]:.2f}" for c in ordered_tickers] 
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
        row=1, cols=2,
        column_widths=[0.82, 0.18],
        subplot_titles=["Pearson correlation (HCA-Ordered)", "Annualized Volatility"],
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
            colorscale="RdBu_r",
            zmin=-1, 
            zmax=1,
            colorscale="RdBu_r",
            colorbar=dict(
                title="Pearson r",
                thickness=15,
                len=0.8,
                x=0.80,
            ),
            # Annotate cells with numeric labels
            texttemplate="%{z:.2f}",
            textfont=dict(size=10)
        ),
        row=1, col=1
    )

    # Volatility bar chart
    vol_ordered = ann_vol[ordered_tickers]
    bar_colors = [
        f"rgba({int(255*(v/vol_ordered.max()))}, 80, 80, 0.85)" for v in vol_ordered.values
    ]

    fig.add_trace(
        go.Bar(
            x=vol_ordered.values,
            y=ordered_tickers,
            orientation="h",
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Ann. Volatility: %{x:.1%}<extra></extra>",
        ),
        row=1, col=2
    )

    # --- Layout tweaks ------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"<b>stock Correlation Heatmap</b> |  |  "
                f"{data_range[0]} to {data_range[1]}<br>"
                f"<sup>Daily log returns  | HCA reordered: Ward linkage, "
                f"d = sqrt(0.5*(1-rho))  | Hover for Spearman r</sup>"
            ),
            x=0.5,
            font=dict(size=14),
        ),
        paper_bgcolor="#0d1117",
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

    # save
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[OK] Interactive heatmap saved -> {output_path}")
    
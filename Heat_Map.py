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
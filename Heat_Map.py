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
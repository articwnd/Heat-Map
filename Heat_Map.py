# Heat_Map.py -- see README.md for full documentation and usage.

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
# from datetime import date

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


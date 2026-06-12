# Stock Correlation Heatmap

Production-grade quantitative tool for visualizing pairwise return correlations
across a configurable equity universe. Outputs an interactive Plotly dashboard
and a publication-ready static clustermap.

---

## Project Structure

```
correlation_heatmap.py   Main pipeline script
requirements.txt         Pinned dependencies
README.md                This file
```

---

## Quickstart

**1. Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**2. Install pinned dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the script**

```bash
python correlation_heatmap.py
```

**Outputs**

| File | Description |
|---|---|
| `correlation_heatmap.html` | Interactive Plotly heatmap, opens in any browser |
| `correlation_heatmap_static.png` | Static clustermap with dendrogram, 150 DPI |

Console also prints a data-quality diagnostic and a correlation summary
that flags high Pearson-Spearman divergence pairs.

---

## Configuration

All user-facing settings live in the `CONFIG` dict at the top of
`correlation_heatmap.py`. No other file needs to be touched.

```python
CONFIG = {
    "tickers":      [...],        # Any valid yfinance ticker strings
    "start_date":   "2022-01-01", # YYYY-MM-DD
    "end_date":     "...",        # Defaults to today
    "min_obs":      126,          # Drop tickers with fewer non-NaN trading days
    "output_html":  "correlation_heatmap.html",
    "output_png":   "correlation_heatmap_static.png",
}
```

---

## Methodology

### Returns
Daily **log returns** are computed as:

```
r_t = ln(P_t / P_{t-1})
```

Log returns are preferred over simple returns because they are time-additive,
more symmetric, and prevent negative compounded values.

### Correlation
Two matrices are computed for every ticker pair using `pandas.DataFrame.corr()`:

| Method | What it measures | Notes |
|---|---|---|
| **Pearson** | Linear co-movement of returns | Sensitive to outliers |
| **Spearman** | Monotonic (rank) co-movement | More robust to fat tails |

A Pearson-Spearman divergence > 0.10 is automatically flagged in the summary
output. Large divergence indicates potential non-linearity or outlier distortion
in the Pearson estimate, both of which matter before feeding the matrix into
an optimizer.

**NaN convention -- pairwise-complete observations.** Both matrices use
pandas' default pairwise-complete handling: for each ticker pair, only the
dates where *both* series have data are used. This keeps Pearson and Spearman
on a consistent footing and means a single late-IPO ticker or missing print
does not NaN out that asset's entire row.

> **Known limitation:** because each pairwise entry may be estimated on a
> different effective sample, the resulting correlation matrix is **not
> guaranteed to be positive semi-definite (PSD)** as a whole, even though
> every individual entry is in `[-1, 1]`. This is fine for the heatmap and
> HCA reorder in this script (the distance metric is computed entrywise).
> It is **not** fine for feeding directly into Markowitz mean-variance
> optimization, which requires a PSD covariance matrix and will error or
> produce nonsensical weights otherwise. If you extend this project toward
> an optimizer, either (a) restrict to `returns.dropna()` for a common
> sample before computing correlations, or (b) apply a PSD-projection
> step (e.g. nearest-correlation-matrix via `statsmodels` or eigenvalue
> clipping) before optimization.

### Hierarchical Clustering (HCA) Reorder
Assets are reordered so that similar assets appear adjacent, revealing block
structure in the heatmap.

**Distance metric:**

```
d(i, j) = sqrt(0.5 * (1 - rho_ij))
```

This is a provably valid metric (satisfies the triangle inequality). It maps:

- `rho = +1` -> `d = 0` (perfectly positively correlated)
- `rho =  0` -> `d = sqrt(0.5) ≈ 0.707` (uncorrelated)
- `rho = -1` -> `d = 1` (perfectly negatively correlated)

**Linkage:** Ward method, which minimizes within-cluster variance at each merge
step. This is the standard choice for financial asset clustering.

### Annualized Volatility
Displayed as a sidebar on the interactive heatmap and as a row color bar on
the static clustermap.

```
ann_vol = daily_std * sqrt(252)
```

---

## Data Source

The default data source is **yfinance**, which pulls adjusted closing prices
from Yahoo Finance. It is free and suitable for research and prototyping.

> **Note:** yfinance accesses Yahoo Finance via unofficial endpoints. It is
> not affiliated with or endorsed by Yahoo, Inc., and is intended for
> personal and educational use. For commercial or production deployments,
> replace `fetch_prices()` with a licensed data provider.

### Swapping to a Production Data Source

The `fetch_prices()` function in `correlation_heatmap.py` is the only place
that needs to change. It must return a `pd.DataFrame` with:
- **Index:** `pd.DatetimeIndex` of trading dates
- **Columns:** ticker strings
- **Values:** adjusted closing prices (float)

**OpenBB (free, open-source, multi-provider)**

```python
from openbb import obb

def fetch_prices(tickers, start, end):
    frames = []
    for t in tickers:
        result = obb.equity.price.historical(
            symbol=t, start_date=start, end_date=end, provider="fmp"
        )
        df = result.to_dataframe()[["close"]].rename(columns={"close": t})
        frames.append(df)
    return pd.concat(frames, axis=1)
```

Install: `pip install openbb openbb-fmp`

**Polygon.io (institutional-grade, paid)**

```python
from polygon import RESTClient

def fetch_prices(tickers, start, end):
    client = RESTClient(api_key="YOUR_KEY")
    frames = []
    for t in tickers:
        bars = client.get_aggs(t, 1, "day", start, end, adjusted=True, limit=50000)
        df = pd.DataFrame(bars)[["timestamp", "close"]].set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_localize(None)
        df = df.rename(columns={"close": t})
        frames.append(df)
    return pd.concat(frames, axis=1)
```

Install: `pip install polygon-api-client`

---

## Requirements

Python >= 3.12 (the script uses built-in generic type hints: `list[str]`,
`tuple[...]`).

See `requirements.txt` for pinned library versions.

---

## Next Steps

This script is the base layer for a broader quant research pipeline:

- **Rolling correlations** -- detect regime shifts over time
- **Markowitz mean-variance optimizer** -- feed the covariance matrix directly
- **Hierarchical Risk Parity (HRP)** -- use the same HCA linkage matrix for
  risk-weighted allocation
- **Stress-period comparison** -- compare calm vs. crisis correlation matrices
  side by side

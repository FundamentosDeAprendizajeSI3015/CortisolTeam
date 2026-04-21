# Exploratory Data Analysis (EDA)

## Objective

Explore the cryptocurrency historical price dataset to understand data distribution,
detect anomalies, and extract insights that guide feature engineering and modeling decisions.

## Setup

```bash
pip install pandas numpy matplotlib seaborn plotly scipy kaleido jupyter
```

> **Note:** `kaleido` is used to export Plotly figures to PNG in headless environments.

## Usage

```bash
# Run the complete EDA script
python src/eda.py

# Or run with Jupyter
jupyter notebook notebooks/02_eda.ipynb
```

## Deliverables

| File | Description |
|------|-------------|
| `src/eda.py` | Main EDA script generating all outputs |
| `reports/eda_summary.md` | Summary of findings and decisions for next steps |
| `reports/figures/01_price_timeseries.png` | Price time series for BTC, ETH, BNB |
| `reports/figures/02_price_normalized.png` | Normalized prices (base=100) |
| `reports/figures/03_volume_timeseries.png` | Trading volume (logarithmic scale) |
| `reports/figures/04_correlation_heatmap.png` | Correlation heatmap of returns |
| `reports/figures/05_volatilidad_rolling.png` | Rolling volatility 30d and 90d |
| `reports/figures/06_outliers_btc.png` | Outliers in BTC closing price |
| `reports/figures/07_outliers_retornos_btc.png` | Outliers in BTC returns |
| `reports/figures/08_hist_qq.png` | Histograms and QQ-plots of returns |
| `reports/figures/09_boxplot_retornos.png` | Comparative boxplot of returns |
| `reports/figures/10_seasonality_mes.png` | Monthly seasonality |
| `reports/figures/11_seasonality_dia.png` | Day-of-week seasonality |
| `reports/figures/12_heatmap_seasonality.png` | Heatmap of returns by month and currency |

## Technical Decisions

- **Data source:** `data/crypto_raw.csv` (pre-validated by `load_data/load_data.py`)
- **`Date` column:** Parsed with `pd.to_datetime()` (arrives as string from CSV)
- **Returns:** `pct_change()` grouped by `Symbol` — NaN in the first record of each currency is expected
- **Rolling volatility:** `min_periods=15` for 30d and `min_periods=45` for 90d to avoid losing data at series start
- **Outliers:** Flagged but **not removed** — used as features in the supervised phase
- **Figure export:** Plotly → `write_image()` with kaleido; Seaborn/Matplotlib → `savefig()` directly
- **Volume visualization:** Logarithmic scale on y-axis to handle wide range of values (from near-zero to billions)

## Dependencies

```
pandas>=2.0.0
numpy
matplotlib
seaborn
plotly
scipy
kaleido
jupyter
nbformat
```

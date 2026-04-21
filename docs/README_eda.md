# Análisis Exploratorio de Datos (EDA)

## Objetivo

Explorar el dataset de precios históricos de criptomonedas para entender la distribución de los datos,
detectar anomalías y extraer insights que guíen las decisiones de feature engineering y modelado.

## Setup

```bash
pip install pandas numpy matplotlib seaborn plotly scipy kaleido==0.2.1 jupyter
```

> **Nota:** Se usa `kaleido==0.2.1` para exportar figuras de Plotly a PNG en entornos sin display.

## Uso

```bash
# Ejecutar el notebook completo
jupyter notebook notebooks/02_eda.ipynb

# O convertir y ejecutar en modo headless
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
```

## Entregables

| Archivo | Descripción |
|---------|-------------|
| `notebooks/02_eda.ipynb` | Notebook ejecutado con outputs incluidos |
| `reports/eda_summary.md` | Resumen de hallazgos y decisiones para el siguiente paso |
| `reports/figures/01_price_timeseries.png` | Serie de precios BTC, ETH, BNB |
| `reports/figures/02_price_normalized.png` | Precios normalizados (base=100) |
| `reports/figures/03_volume_timeseries.png` | Volumen de trading |
| `reports/figures/04_correlation_heatmap.png` | Heatmap de correlación de retornos |
| `reports/figures/05_volatilidad_rolling.png` | Volatilidad rolling 30d y 90d |
| `reports/figures/06_outliers_btc.png` | Outliers en precio de BTC |
| `reports/figures/07_outliers_retornos_btc.png` | Outliers en retornos de BTC |
| `reports/figures/08_hist_qq.png` | Histogramas y QQ-plots de retornos |
| `reports/figures/09_boxplot_retornos.png` | Boxplot comparativo de retornos |
| `reports/figures/10_seasonality_mes.png` | Estacionalidad mensual |
| `reports/figures/11_seasonality_dia.png` | Estacionalidad por día de semana |
| `reports/figures/12_heatmap_seasonality.png` | Heatmap de retorno por mes y moneda |

## Decisiones técnicas

- **Fuente de datos:** `data/crypto_raw.csv` (ya validado por `load_data/load_data.py`)
- **Columna `Date`:** se parsea con `pd.to_datetime()` (llega como string del CSV)
- **Retornos:** `pct_change()` agrupado por `Symbol` — NaN en el primer registro de cada moneda es correcto
- **Volatilidad rolling:** `min_periods=15` para 30d y `min_periods=45` para 90d para no perder el inicio de series cortas
- **Outliers:** Se marcan pero **no se eliminan** — se usan como features en la fase supervisada
- **Exportación de figuras:** Plotly → `write_image()` con kaleido 0.2.1; Seaborn/Matplotlib → `savefig()` directamente
- **Stack de correlaciones:** Se renombran los niveles del índice antes de `stack()` para evitar conflictos de columnas duplicadas con pandas ≥ 2.0

## Dependencias

```
pandas>=2.0.0
numpy
matplotlib
seaborn
plotly
scipy
kaleido==0.2.1
jupyter
nbformat
```

# Visualizacion Interactiva - Crypto ML Project

Modulo para generar visualizaciones de EDA y clustering, y construir un dashboard HTML interactivo.

## Resumen de entrega (version corta)

- Scripts del modulo: `app/visualize.py`, `app/generate_dashboard.py`, `app/main.py`
- Salida principal: `visualization/app/dashboard.html` y figuras en `visualization/outputs/`
- Total de visualizaciones activas: 13 PNG
- Datos de entrada: `data/crypto_raw.csv`, `data/features_clustering.csv`, `data/cluster_labels.csv`

## Graficas generadas y que muestra cada una

### EDA

- `01_price_timeseries.png`: serie historica de precios de cierre para BTC, ETH y BNB.
- `02_price_normalized.png`: compara crecimiento relativo entre monedas con base 100.
- `03_returns_distribution.png`: distribucion de retornos diarios con curva normal de referencia.
- `04_correlation_heatmap.png`: correlacion de retornos entre criptomonedas para ver dependencias.
- `05_volume_analysis.png`: volumen de trading por moneda en escala logaritmica.
- `09_rolling_volatility.png`: volatilidad rodante (30d y 90d) para medir cambios de riesgo.
- `10_seasonality_month.png`: retorno promedio por mes para detectar estacionalidad mensual.
- `11_seasonality_dayofweek.png`: retorno promedio por dia de semana para patrones semanales.
- `12_outliers_btc_returns.png`: outliers en retornos de BTC usando Z-Score e IQR.

### No supervisado

- `13_k_selection_metrics.png`: apoyo para elegir K con codo, silhouette y Davies-Bouldin.
- `06_pca_clustering.png`: proyeccion PCA 2D de clusters para varios algoritmos.
- `07_features_heatmap.png`: mapa de calor de features normalizadas por moneda y cluster.
- `08_cluster_statistics.png`: boxplots de features para comparar perfiles entre clusters.

## Requisitos

Instalar dependencias desde la raiz del proyecto:

```bash
pip install -r requirements.txt
```

Paquetes usados: pandas, numpy, matplotlib, seaborn, scikit-learn.

## Ejecucion

### Opcion recomendada

```bash
python visualization/app/main.py
```


### Archivos del modulo

- `app/visualize.py`: genera las figuras PNG y valida datos de entrada.
- `app/generate_dashboard.py`: construye el dashboard HTML.
- `app/main.py`: orquesta ejecucion completa (figuras + dashboard).
- `README.md`: documentacion consolidada del modulo.

### Dashboard interactivo

- Archivo: `visualization/app/dashboard.html`
- Navegacion por secciones: EDA, No Supervisado y Supervisado (placeholder).
- Incluye tarjetas con descripcion y fecha de actualizacion automatica.
- Compatible con escritorio y movil.

Funciones EDA:

- `plot_price_timeseries()`
- `plot_price_normalized()`
- `plot_returns_distribution()`
- `plot_correlation_heatmap()`
- `plot_volume_analysis()`
- `plot_rolling_volatility()`
- `plot_seasonality_month()`
- `plot_seasonality_dayofweek()`
- `plot_outliers_detection()`

Funciones No Supervisado:

- `plot_k_selection_metrics()`
- `plot_pca_clustering()`
- `plot_features_heatmap()`
- `plot_cluster_statistics()`


### Licencia

Uso libre para fines educativos e investigacion.

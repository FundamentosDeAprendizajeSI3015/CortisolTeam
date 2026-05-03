# Visualizacion Interactiva - Crypto ML Project

Modulo para generar visualizaciones de EDA y clustering, y construir un dashboard HTML interactivo.

## Resumen de entrega (version corta)

- Scripts del modulo: `app/visualize.py`, `app/generate_dashboard.py`, `app/main.py`
- Salida principal: `visualization/app/dashboard.html` y figuras en `visualization/outputs/`
- Total de visualizaciones activas: 13 PNG (EDA + clustering) + figuras supervisadas y de scoring
- Datos de entrada: `data/crypto_raw.csv`, `data/features_clustering.csv`, `data/cluster_labels.csv`
- Fuentes adicionales: `supervised/reports/figures/`, `scoring/reports/figures/`, `scoring/reports/metrics_summary.csv`, `scoring/reports/model_report.pdf`

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

### Supervisado

- `13_comparacion_clasificadores.png`: AUC val vs test por clasificador.
- `14_comparacion_regresores.png`: R2 val vs test por regresor.
- `15_confusion_matrix.png`: matriz de confusion del mejor clasificador.
- `16_roc_curve.png`: curva ROC del mejor clasificador.
- `17_feature_importance_ran.png`: importancia de features del clasificador.
- `18_feature_importance_xgb.png`: importancia de features del regresor.
- `19_predicciones_vs_real.png`: predicho vs real para el regresor.
- `20_residuos.png`: distribucion y dispersion de residuos.
- `21_3d_features_vs_return.png`: relacion 3D entre features y return.
- `22_3d_predicciones.png`: real vs predicho vs residuo en 3D.
- `24_heatmap_correlacion.png`: correlacion entre features y lags.
- `25_boxplot_return_por_simbolo.png`: distribucion de return por simbolo.
- `26_curva_aprendizaje_log.png`: curva de aprendizaje del clasificador.
- `27_curva_aprendizaje_xgb.png`: curva de aprendizaje del regresor.
- `28_tabla_metricas.png`: tabla comparativa de metricas.
- `29_scatter_matrix.png`: matriz de dispersion de features.

### Scoring

- `17_scoring_clf_comparison.png`: comparacion final de clasificadores.
- `18_scoring_confusion_matrices.png`: matrices de confusion.
- `19_scoring_roc_curves.png`: curvas ROC comparativas.
- `20_scoring_regression_scatter.png`: predicho vs real en regresion.
- `21_scoring_residuals.png`: diagnostico de residuos.
- `22_scoring_clustering_metrics.png`: metricas de clustering.
- `23_scoring_backtest.png`: backtesting de estrategia vs buy & hold.

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
- Navegacion por secciones: EDA, No Supervisado, Supervisado y Scoring.
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


### Uso de IA
Esta visualización fue desarrollada con apoyo de herramientas de inteligencia artificial (Claude, Anthropic).

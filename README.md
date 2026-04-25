# 🏆 [06] Scoring — Crypto ML Project

## Objetivo
Evaluar formalmente todos los modelos (clasificación, regresión, clustering) con métricas estándar, ejecutar backtesting y generar el reporte final consolidado.

## Setup

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Run feature engineering first (if features.parquet doesn't exist)
python src/feature_engineering.py

# Run the complete scoring pipeline
python src/scoring.py

# Or open the interactive notebook
# notebooks/06_scoring.ipynb
```

## Entregables

- `src/scoring.py` — Script ejecutable del pipeline completo de scoring
- `notebooks/06_scoring.ipynb` — Notebook documentado con análisis paso a paso
- `reports/metrics_summary.csv` — CSV consolidado con todas las métricas
- `reports/model_report.pdf` — Reporte final en PDF con tablas y figuras
- `reports/figures/17_*` a `23_*` — 7 figuras generadas por el scoring
- `models/best_classifier.pkl` — Mejor clasificador serializado
- `models/best_regressor.pkl` — Regresor serializado

## Estructura

```
├── data/
│   ├── crypto_raw.csv                  # Dataset original
│   ├── processed/features.parquet      # Features con indicadores técnicos
│   ├── features_clustering.csv         # Features para clustering
│   └── cluster_labels.csv              # Etiquetas de clustering
├── models/
│   ├── best_classifier.pkl             # RandomForestClassifier
│   └── best_regressor.pkl              # RandomForestRegressor
├── notebooks/
│   └── 06_scoring.ipynb                # Notebook interactivo
├── reports/
│   ├── metrics_summary.csv             # Métricas consolidadas
│   ├── model_report.pdf                # Reporte PDF final
│   └── figures/
│       ├── 17_scoring_clf_comparison.png
│       ├── 18_scoring_confusion_matrices.png
│       ├── 19_scoring_roc_curves.png
│       ├── 20_scoring_regression_scatter.png
│       ├── 21_scoring_residuals.png
│       ├── 22_scoring_clustering_metrics.png
│       └── 23_scoring_backtest.png
├── src/
│   ├── scoring.py                      # Script principal de scoring
│   ├── feature_engineering.py          # Pipeline de features (desde supervised)
│   └── supervised.py                   # Entrenamiento supervisado (referencia)
├── unsupervised/
│   ├── clustering.py                   # Clustering (referencia)
│   └── eda_clustering.py               # EDA clustering (referencia)
└── requirements.txt
```

## Decisiones técnicas

### Clasificación
- Se evaluaron **3 modelos**: LogisticRegression, RandomForestClassifier, XGBClassifier
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- Split temporal 80/10/10 con TimeSeriesSplit (5 folds) para cross-validation
- Los modelos rondan AUC ≈ 0.50, consistente con la dificultad inherente de predecir dirección diaria de precios crypto

### Regresión
- RandomForestRegressor para predicción de precio de cierre
- Métricas: MAE, RMSE, MAPE, R²
- R² alto en validación (0.998) pero menor en test (0.46) — refleja la explosión de precios 2020-2021

### Clustering
- Re-evaluación de KMeans (K=2, K=4), DBSCAN, Agglomerative
- Métricas: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- KMeans K=2 obtiene Silhouette = 0.67 (buena separación)

### Backtesting
- Estrategia simple: comprar cuando el clasificador predice subida, vender cuando predice bajada
- Comparación contra buy & hold para BTC y ETH
- La estrategia underperforma buy & hold durante el bull market 2020-2021

### Reporte PDF
- Generado con `reportlab` — incluye tablas formateadas y todas las figuras
- Secciones: Clasificación, Regresión, Clustering, Backtesting, Resumen consolidado

## Dependencias
- `feature/supervised` — modelos supervisados, features.parquet
- `feature/unsupervised` — clustering, features_clustering.csv

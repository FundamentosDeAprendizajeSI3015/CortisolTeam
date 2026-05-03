# Scoring

## Objetivo
Evaluar formalmente todos los modelos con metricas estandar y generar el reporte final.

## Dataset / Contexto
Esta rama consume los datos producidos por la etapa anterior.
Depende de: feature/supervised + feature/unsupervised.

## Tareas
- Clasificacion: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Regresion: MAE, RMSE, MAPE, R2
- Clustering: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
- Comparar todos los modelos en una tabla consolidada
- Backtesting simple: simular estrategia buy/sell basada en predicciones vs buy & hold
- Generar reporte final en reports/model_report.pdf

## Entregables
| Archivo / Artefacto | Descripcion |
|---|---|
| scoring/src/scoring.py | Script Python ejecutable |
| scoring/reports/model_report.pdf | Reporte final exportado |
| scoring/reports/metrics_summary.csv | Dataset procesado |

## Dependencias de rama
Depende de: feature/supervised + feature/unsupervised

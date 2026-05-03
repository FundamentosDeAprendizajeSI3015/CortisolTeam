# Fase 04 — Modelado Supervisado

Pipeline completo de clasificación y regresión sobre features técnicos de criptomonedas.
Predice si el precio sube o baja al día siguiente (clasificación) y el retorno porcentual diario (regresión).

---

## Modelos implementados

### Clasificadores — target: sube/baja precio (binario)
| Modelo | Descripción |
|--------|-------------|
| `LogisticRegression` | Modelo lineal con regularización L2, escalado estándar incluido |
| `RandomForestClassifier` | Ensemble de árboles con balanceo de clases |
| `XGBClassifier` | Gradient boosting optimizado para clasificación binaria |

### Regresores — target: retorno porcentual del día siguiente
| Modelo | Descripción |
|--------|-------------|
| `Ridge` | Regresión lineal con regularización L2, escalado estándar incluido |
| `RandomForestRegressor` | Ensemble de árboles para regresión continua |
| `XGBRegressor` | Gradient boosting para regresión, robusto ante no-linealidades |

---

## Decisiones técnicas

### Features de lag
Se agregan features con valores desplazados t-1, t-3 y t-7 días sobre `Return`, `rsi_14` y `macd`, agrupando por símbolo para no mezclar series temporales. Esto le da memoria temporal a los modelos sin usar arquitecturas secuenciales. El total de features pasa de 11 base a 20 con lags.

### Filtro de outliers en Return
La columna `Return` presenta valores extremos (hasta 9.7B) producto de errores en datos históricos de monedas de bajo volumen. Se filtran filas fuera del rango `[-1.0, +5.0]` (-100% a +500% diario), eliminando solo 11 filas de 32,274.

### Target del regresor
Se usa `Return` (retorno porcentual diario) en lugar de `Close` (precio de cierre) porque `Return` es una serie estacionaria con sentido financiero real. Predecir `Close` con modelos de árbol produce R² artificialmente altos sin valor predictivo real.

### Exclusión de data leakage
En el regresor, `Return` y sus lags son excluidos de las features porque son el mismo target. Solo se usan lags de `rsi_14` y `macd` como memoria temporal.

### Validación temporal
Se usa `TimeSeriesSplit` con 5 folds en lugar de KFold estándar para respetar el orden cronológico y evitar que datos futuros contaminen el entrenamiento.

### Tuning de hiperparámetros
Se usa `RandomizedSearchCV` con 20 iteraciones por modelo, optimizando ROC-AUC para clasificadores y R² para regresores.

### Split temporal
```
Train: 80%  (2013-06-04 → 2019-11-22)  21,476 filas
Val:   10%  (2019-11-23 → 2020-09-12)   4,839 filas
Test:  10%  (2020-09-13 → 2021-07-05)   5,808 filas
```

---

## Métricas obtenidas

### Clasificadores

| Modelo | CV_AUC | Val_Acc | Val_F1 | Val_AUC | Test_Acc | Test_F1 | Test_AUC |
|--------|--------|---------|--------|---------|----------|---------|----------|
| LogisticRegression | 0.5353 | 0.5392 | 0.5688 | 0.5624 | 0.5122 | 0.5382 | 0.5280 |
| RandomForestClassifier | 0.5569 | 0.5292 | 0.5291 | 0.5404 | 0.5112 | 0.5038 | 0.5173 |
| XGBClassifier | 0.5620 | 0.5158 | 0.4965 | 0.5319 | 0.5176 | 0.4902 | 0.5248 |

**Mejor clasificador:** LogisticRegression (Val AUC = 0.5624)

> AUC > 0.5 indica señal real capturada. Predecir movimientos de crypto es intrínsecamente difícil por la alta volatilidad y eficiencia del mercado.

### Regresores

| Modelo | CV_R² | Val_MAE | Val_RMSE | Val_R² | Test_MAE | Test_RMSE | Test_R² |
|--------|-------|---------|----------|--------|----------|-----------|---------|
| Ridge | -8.4073 | 0.020141 | 0.030137 | 0.6871 | 0.023124 | 0.056915 | 0.6277 |
| RandomForestRegressor | 0.7381 | 0.010899 | 0.022759 | 0.8216 | 0.012582 | 0.045984 | 0.7570 |
| XGBRegressor | 0.7133 | 0.010710 | 0.021431 | 0.8418 | 0.014143 | 0.045762 | 0.7593 |

**Mejor regresor:** XGBRegressor (Val R² = 0.8418, Test R² = 0.7593)

> R² positivo en test confirma que el modelo generaliza. Los lags de indicadores técnicos son los features más importantes según feature importance.

---

## Uso

```bash
python supervised/supervised.py
```

### Prerequisitos
```bash
pip install -r requirements.txt
pip install xgboost scikit-learn joblib matplotlib seaborn pyarrow
```

### Input
```
data/processed/features.parquet     # generado por feature_engineering.py
<<<<<<< HEAD
=======
data/cluster_labels.csv             # generado por unsupervised/clustering.py
data/features_clustering.csv        # generado por unsupervised/clustering.py
>>>>>>> origin/desarrollo
```

### Outputs
```
<<<<<<< HEAD
supervised/models/best_classifier.pkl
supervised/models/best_regressor.pkl
supervised/reports/supervised_report.md
supervised/reports/figures/13_comparacion_clasificadores.png
supervised/reports/figures/14_comparacion_regresores.png
supervised/reports/figures/15_confusion_matrix.png
supervised/reports/figures/16_roc_curve.png
supervised/reports/figures/17_feature_importance_clf.png
supervised/reports/figures/18_feature_importance_reg.png
supervised/reports/figures/19_predicciones_vs_real.png
supervised/reports/figures/20_residuos.png
supervised/reports/figures/21_3d_features_vs_return.png
supervised/reports/figures/22_3d_predicciones.png
supervised/reports/figures/24_heatmap_correlacion.png
supervised/reports/figures/25_boxplot_return_por_simbolo.png
supervised/reports/figures/26_curva_aprendizaje_clf.png
supervised/reports/figures/27_curva_aprendizaje_reg.png
supervised/reports/figures/28_tabla_metricas.png
supervised/reports/figures/29_scatter_matrix.png
=======
models/best_classifier.pkl
models/best_regressor.pkl
reports/supervised_report.md
reports/figures/13_comparacion_clasificadores.png
reports/figures/14_comparacion_regresores.png
reports/figures/15_confusion_matrix.png
reports/figures/16_roc_curve.png
reports/figures/17_feature_importance_clf.png
reports/figures/18_feature_importance_reg.png
reports/figures/19_predicciones_vs_real.png
reports/figures/20_residuos.png
reports/figures/21_3d_features_vs_return.png
reports/figures/22_3d_predicciones.png
reports/figures/23_3d_pca_clusters.png
reports/figures/24_heatmap_correlacion.png
reports/figures/25_boxplot_return_por_simbolo.png
reports/figures/26_curva_aprendizaje_clf.png
reports/figures/27_curva_aprendizaje_reg.png
reports/figures/28_tabla_metricas.png
reports/figures/29_scatter_matrix.png
>>>>>>> origin/desarrollo
```

---

## Figuras generadas

### Gráficas 2D
| Figura | Descripción |
|--------|-------------|
| `13_comparacion_clasificadores.png` | AUC val vs test por clasificador |
| `14_comparacion_regresores.png` | R² val vs test por regresor |
| `15_confusion_matrix.png` | Matriz de confusión del mejor clasificador |
| `16_roc_curve.png` | Curva ROC del mejor clasificador |
| `17_feature_importance_clf.png` | Importancia de features del clasificador |
| `18_feature_importance_reg.png` | Importancia de features del regresor |
| `19_predicciones_vs_real.png` | Scatter predicciones vs valores reales |
| `20_residuos.png` | Histograma y scatter de residuos |
| `24_heatmap_correlacion.png` | Correlación entre todas las features + lags |
| `25_boxplot_return_por_simbolo.png` | Distribución de Return por criptomoneda |
| `26_curva_aprendizaje_clf.png` | Curva de aprendizaje del mejor clasificador |
| `27_curva_aprendizaje_reg.png` | Curva de aprendizaje del mejor regresor |
| `28_tabla_metricas.png` | Tabla visual comparativa de métricas |
| `29_scatter_matrix.png` | Scatter matrix de features principales |

### Gráficas 3D
| Figura | Descripción |
|--------|-------------|
| `21_3d_features_vs_return.png` | SMA_7 vs RSI_14 vs MACD coloreado por Return |
| `22_3d_predicciones.png` | Return real vs predicho vs residuo en 3D |
<<<<<<< HEAD
=======
| `23_3d_pca_clusters.png` | PCA 3D de criptomonedas coloreado por cluster KMeans K=4 |
>>>>>>> origin/desarrollo

---

## Estructura de archivos

```
supervised/
<<<<<<< HEAD
├── feature_engineering.py
├── supervised.py
├── models/
├── reports/
│   └── figures/
└── README.md
=======
├── supervised.py     # pipeline completo
└── README.md         # este archivo
>>>>>>> origin/desarrollo
```
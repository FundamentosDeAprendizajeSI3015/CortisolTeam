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
Se agregan features con valores desplazados t-1, t-3 y t-7 días sobre `Return`, `rsi_14` y `macd`, agrupando por símbolo para no mezclar series temporales. Esto le da memoria temporal a los modelos sin usar arquitecturas secuenciales.

### Filtro de outliers en Return
La columna `Return` del dataset crudo presenta valores extremos (hasta 9.7B) producto de errores en datos históricos de monedas de bajo volumen. Se filtran filas fuera del rango `[-1.0, +5.0]` (-100% a +500% diario), eliminando solo 11 filas de 32,274.

### Target del regresor
Se usa `Return` (retorno porcentual diario) en lugar de `Close` (precio de cierre) porque `Return` es una serie estacionaria con sentido financiero real. Predecir `Close` directamente con modelos de árbol produce R² artificialmente altos sin valor predictivo real.

### Exclusión de data leakage
En el regresor, `Return` y sus lags son excluidos de las features porque son el mismo target. Solo se usan lags de `rsi_14` y `macd` como memoria temporal.

### Validación temporal
Se usa `TimeSeriesSplit` con 5 folds en lugar de KFold estándar para respetar el orden cronológico y evitar que datos futuros contaminen el entrenamiento.

### Tuning de hiperparámetros
Se usa `RandomizedSearchCV` con 20 iteraciones por modelo, optimizando ROC-AUC para clasificadores y R² para regresores.

### Split temporal
```
Train: 80%  (2013-06-04 → 2019-11-22)
Val:   10%  (2019-11-23 → 2020-09-12)
Test:  10%  (2020-09-13 → 2021-07-05)
```

---

## Métricas obtenidas

### Clasificadores

| Modelo | CV_AUC | Val_AUC | Test_AUC |
|--------|--------|---------|----------|
| LogisticRegression | 0.5353 | 0.5624 | 0.5280 |
| RandomForestClassifier | 0.5569 | 0.5404 | 0.5173 |
| XGBClassifier | 0.5620 | 0.5319 | 0.5248 |

**Mejor clasificador:** LogisticRegression (Val AUC = 0.5624)

> AUC > 0.5 indica señal real capturada. Predecir movimientos de crypto es intrínsecamente difícil por la alta volatilidad y eficiencia del mercado.

### Regresores

| Modelo | CV_R² | Val_R² | Test_R² | Test_MAE |
|--------|-------|--------|---------|----------|
| Ridge | -8.4073 | 0.6871 | 0.6277 | 0.023124 |
| RandomForestRegressor | 0.7381 | 0.8216 | 0.7570 | 0.012582 |
| XGBRegressor | 0.7133 | 0.8418 | 0.7593 | 0.014143 |

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
data/processed/features.parquet   # generado por feature_engineering.py
```

### Outputs
```
models/best_classifier.pkl                        # mejor clasificador serializado
models/best_regressor.pkl                         # mejor regresor serializado
reports/supervised_report.md                      # reporte completo de métricas
reports/figures/13_comparacion_clasificadores.png
reports/figures/14_comparacion_regresores.png
reports/figures/15_confusion_matrix.png
reports/figures/16_roc_curve.png
reports/figures/17_feature_importance_clf.png
reports/figures/18_feature_importance_reg.png
reports/figures/19_predicciones_vs_real.png
reports/figures/20_residuos.png
```

---

## Estructura de archivos

```
supervised/
├── supervised.py     # pipeline completo
└── README.md         # este archivo
```
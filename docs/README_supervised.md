# Supervisado — Crypto ML Project

## Objetivo

Predecir el movimiento de precio del día siguiente (clasificación binaria) y el precio de cierre
(regresión) usando indicadores técnicos calculados sobre el dataset histórico de criptomonedas.

## Setup

```bash
pip install pandas pyarrow numpy matplotlib seaborn scikit-learn xgboost ta joblib
```

## Uso

```bash
# 1. Generar features (requiere data/crypto_raw.csv)
python src/feature_engineering.py

# 2. Entrenar modelos y evaluar
python src/supervised.py
```

## Entregables

| Archivo | Descripción |
|---------|-------------|
| `data/processed/features.parquet` | Dataset con indicadores técnicos y target |
| `models/best_classifier.pkl` | Mejor clasificador (RandomForestClassifier) |
| `models/best_regressor.pkl` | Regresor de precio de cierre (RandomForestRegressor) |
| `reports/figures/13_comparacion_modelos.png` | AUC comparativo en Val y Test |
| `reports/figures/14_confusion_matrix.png` | Matriz de confusión del mejor clasificador |
| `reports/figures/15_roc_curve.png` | Curva ROC del mejor clasificador |
| `reports/figures/16_feature_importance.png` | Importancia de features |

## Decisiones técnicas

### Feature Engineering (`src/feature_engineering.py`)
- **Monedas excluidas:** USDT, USDC (stablecoins sin señal de precio), WBTC (colineal con BTC, r=0.93)
- **Indicadores:** SMA 7/14/30, EMA 14, RSI 14, Bollinger Bands 20d, MACD — calculados con librería `ta`
- **Target clasificación:** `target = 1` si `Close(t+1) > Close(t)`, `0` si no. NaN en el último registro de cada símbolo.
- **NaN:** Se eliminan filas donde `sma_30`, `rsi_14`, `macd` o `target` son NaN (primeros ~30 días de cada símbolo).

### Modelado (`src/supervised.py`)
- **Split temporal 80/10/10** sobre fechas globales, sin shuffle — preserva la estructura temporal.
  - Train: 2013-06-01 → 2019-11-21 (21,519 filas)
  - Val:   2019-11-22 → 2020-09-12  (4,858 filas)
  - Test:  2020-09-13 → 2021-07-05  (5,817 filas)
- **Validación cruzada:** `TimeSeriesSplit(n_splits=5)` sobre el set de entrenamiento.
- **Selección de mejor clasificador:** por `Val_AUC`.
- **Nota sobre el regresor:** El CV_R2 es negativo porque el modelo se entrena en precios de 2013-2019 (rangos bajos) y la validación cruzada evalúa en sub-períodos de ese mismo rango. El `Test_R2 ≈ 0.46` es el número relevante. Para producción se recomienda predecir retornos en lugar de precios absolutos.
- **`random_state=42`** en todos los modelos para reproducibilidad.

## Resultados

| Modelo | CV AUC | Val AUC | Test AUC |
|--------|:------:|:-------:|:--------:|
| LogisticRegression | 0.509 | 0.489 | 0.477 |
| **RandomForestClassifier** | **0.536** | **0.559** | **0.524** |
| XGBClassifier | 0.530 | 0.551 | 0.533 |

Los AUC cercanos a 0.5 son consistentes con la hipótesis de mercado eficiente: predecir la
dirección del precio a corto plazo es difícil. RandomForest gana marginalmente en validación.

## Dependencias

```
pandas>=2.0.0
pyarrow
numpy
matplotlib
seaborn
scikit-learn
xgboost
ta
joblib
```

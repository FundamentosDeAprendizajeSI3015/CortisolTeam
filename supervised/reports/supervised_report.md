# Reporte de Modelado Supervisado — Crypto ML Project

**Generado:** 2026-05-17 17:38

---

## 1. Configuración del experimento
- **Features base:** 16 (SMA, EMA, RSI, BB, MACD, ATR, Stochastic, Williams %R, OBV)
- **Features con lags (t-1, t-3, t-7):** 28
- **Split temporal:** Train 684981 filas | Val 142000 filas | Test 159920 filas
- **Validación cruzada:** TimeSeriesSplit (5 folds)
- **Tuning:** RandomizedSearchCV (10 iteraciones, espacio ampliado)
- **Filtro outliers Return:** [-1.0, 5.0]

---

## 2. Clasificadores — predice si el precio sube o baja

### Métricas comparativas

| Modelo | CV_AUC | Val_Acc | Val_F1 | Val_AUC | Test_Acc | Test_F1 | Test_AUC |
|--------|--------|---------|--------|---------|----------|---------|----------|
| LogisticRegression | 0.5144 | 0.5303 | 0.5234 | 0.5497 | 0.5445 | 0.5228 | 0.5681 |
| RandomForestClassifier | 0.537 | 0.5376 | 0.6506 | 0.5708 | 0.5485 | 0.6476 | 0.591 |
| XGBClassifier | 0.5356 | 0.5378 | 0.4386 | 0.5699 | 0.5498 | 0.441 | 0.5894 |

### Mejor clasificador: **RandomForestClassifier**
- Val AUC: 0.5708
- Test AUC: 0.591
- Hiperparámetros: `{'clf__n_estimators': 100, 'clf__min_samples_leaf': 3, 'clf__max_features': 'log2', 'clf__max_depth': 5}`

### Interpretación
- AUC > 0.5 indica que los modelos capturan alguna señal real en los datos.
- Predecir movimientos de criptomonedas es difícil por la alta volatilidad.
- Los features de lag aportan memoria temporal que mejora la capacidad predictiva.

---

## 3. Regresores — predice el retorno porcentual del día siguiente

### Métricas comparativas

| Modelo | CV_MAE | Val_MAE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 |
|--------|-------|---------|----------|--------|----------|-----------|---------|
| Ridge | 39332.579093 | 0.017183 | 0.027996 | 0.6763 | 0.01644 | 0.02631 | 0.6715 |
| RandomForestRegressor | 0.011622 | 0.008924 | 0.016575 | 0.8865 | 0.00836 | 0.015205 | 0.8903 |
| XGBRegressor | 0.009417 | 0.007965 | 0.014025 | 0.9188 | 0.007599 | 0.01331 | 0.9159 |

### Mejor regresor: **XGBRegressor**
- Val R²: 0.9188
- Test R²: 0.9159
- Hiperparámetros: `{'reg__subsample': 0.8, 'reg__n_estimators': 200, 'reg__min_child_weight': 1, 'reg__max_depth': 7, 'reg__learning_rate': 0.1, 'reg__colsample_bytree': 0.8}`

### Interpretación
- El target es el retorno porcentual diario — variable estacionaria.
- R² positivo en test indica que el modelo generaliza más allá del azar.
- Return fue excluido de las features del regresor para evitar data leakage.
- Los lags de rsi_14 y macd capturan inercia de los indicadores técnicos.

---

## 4. Figuras generadas
### Gráficas 2D
- `13_comparacion_clasificadores.png` — AUC val vs test por modelo
- `14_comparacion_regresores.png` — R² val vs test por modelo
- `15_confusion_matrix.png` — Matriz de confusión del mejor clasificador
- `16_roc_curve.png` — Curva ROC del mejor clasificador
- `30_precision_recall.png` — Curva Precision-Recall con Average Precision y baseline
- `17_feature_importance_clf.png` — Importancia de features (clasificador)
- `18_feature_importance_reg.png` — Importancia de features (regresor)
- `19_predicciones_vs_real.png` — Scatter predicciones vs valores reales
- `20_residuos.png` — Distribución de residuos del mejor regresor
- `24_heatmap_correlacion.png` — Correlación entre features
- `25_boxplot_return_por_simbolo.png` — Distribución de Return por moneda
- `26_curva_aprendizaje_clf.png` — Curva de aprendizaje del clasificador
- `27_curva_aprendizaje_reg.png` — Curva de aprendizaje del regresor
- `28_tabla_metricas.png` — Tabla visual de métricas comparativas
- `29_scatter_matrix.png` — Scatter matrix de features principales
### Gráficas 3D
- `21_3d_features_vs_return.png` — Features vs Return en espacio 3D
- `22_3d_predicciones.png` — Real vs Predicho vs Residuo en 3D
- `23_3d_pca_clusters.png` — PCA 3D coloreado por cluster KMeans
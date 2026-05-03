# Reporte de Modelado Supervisado — Crypto ML Project

**Generado:** 2026-05-03 02:54

---

## 1. Configuración del experimento
- **Features base:** 11
- **Features con lags (t-1, t-3, t-7):** 20
- **Split temporal:** Train 21476 filas | Val 4839 filas | Test 5808 filas
- **Validación cruzada:** TimeSeriesSplit (5 folds)
- **Tuning:** RandomizedSearchCV (20 iteraciones)
- **Filtro outliers Return:** [-1.0, 5.0]

---

## 2. Clasificadores — predice si el precio sube o baja

### Métricas comparativas

| Modelo | CV_AUC | Val_Acc | Val_F1 | Val_AUC | Test_Acc | Test_F1 | Test_AUC |
|--------|--------|---------|--------|---------|----------|---------|----------|
| LogisticRegression | 0.5353 | 0.5392 | 0.5688 | 0.5624 | 0.5122 | 0.5382 | 0.528 |
| RandomForestClassifier | 0.5569 | 0.5292 | 0.5291 | 0.5404 | 0.5112 | 0.5038 | 0.5173 |
| XGBClassifier | 0.562 | 0.5111 | 0.4916 | 0.5286 | 0.5157 | 0.4834 | 0.5226 |

### Mejor clasificador: **LogisticRegression**
- Val AUC: 0.5624
- Test AUC: 0.528
- Hiperparámetros: `{'clf__solver': 'lbfgs', 'clf__C': 1.0}`

### Interpretación
- AUC > 0.5 indica que los modelos capturan alguna señal real en los datos.
- Predecir movimientos de criptomonedas es difícil por la alta volatilidad.
- Los features de lag aportan memoria temporal que mejora la capacidad predictiva.

---

## 3. Regresores — predice el retorno porcentual del día siguiente

### Métricas comparativas

| Modelo | CV_R2 | Val_MAE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 |
|--------|-------|---------|----------|--------|----------|-----------|---------|
| Ridge | -8.4073 | 0.020141 | 0.030137 | 0.6871 | 0.023124 | 0.056915 | 0.6277 |
| RandomForestRegressor | 0.7381 | 0.010899 | 0.022759 | 0.8216 | 0.012582 | 0.045984 | 0.757 |
| XGBRegressor | 0.7133 | 0.010571 | 0.021028 | 0.8477 | 0.013784 | 0.044281 | 0.7746 |

### Mejor regresor: **XGBRegressor**
- Val R²: 0.8477
- Test R²: 0.7746
- Hiperparámetros: `{'reg__subsample': 0.8, 'reg__n_estimators': 200, 'reg__max_depth': 7, 'reg__learning_rate': 0.05}`

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
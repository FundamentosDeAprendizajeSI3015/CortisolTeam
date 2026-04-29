"""
[04] Supervisado — Crypto ML Project
Entrena clasificadores y regresores sobre features técnicos de criptomonedas.

Clasificadores (target: sube/baja precio al día siguiente):
    - LogisticRegression
    - RandomForestClassifier
    - XGBClassifier

Regresores (target: retorno porcentual del día siguiente):
    - Ridge
    - RandomForestRegressor
    - XGBRegressor

Mejoras aplicadas:
    - Features de lag (t-1, t-3, t-7) para memoria temporal
    - RandomizedSearchCV para tuning de hiperparámetros
    - Reporte completo en reports/supervised_report.md
    - Gráfico predicciones vs real del mejor regresor
    - Gráfico de residuos del mejor regresor

Uso:
    python src/supervised.py

Salidas:
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
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data' / 'processed' / 'features.parquet'
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'reports' / 'figures'
REPORTS_DIR = ROOT / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURE_COLS = [
    'sma_7', 'sma_14', 'sma_30',
    'ema_14', 'rsi_14',
    'bb_high', 'bb_low', 'bb_width',
    'macd', 'macd_signal',
    'Return',
]
LAG_COLS_SOURCE = ['Return', 'rsi_14', 'macd']
LAG_PERIODS     = [1, 3, 7]

TARGET_CLF   = 'target'
TARGET_REG   = 'Return'
RETURN_MIN   = -1.0
RETURN_MAX   =  5.0
RANDOM_STATE = 42
N_ITER_SEARCH = 20


# ── Lag features ───────────────────────────────────────────────────────────

def agregar_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de lag por símbolo para dar memoria temporal al modelo.

    Para cada columna en LAG_COLS_SOURCE y cada período en LAG_PERIODS,
    crea una columna con el valor desplazado t períodos atrás,
    agrupando por símbolo para no mezclar series temporales.

    Args:
        df: DataFrame con columna Symbol y columnas base.

    Returns:
        DataFrame con columnas de lag agregadas.
    """
    df = df.copy()
    for col in LAG_COLS_SOURCE:
        for lag in LAG_PERIODS:
            df[f'{col}_lag{lag}'] = df.groupby('Symbol')[col].shift(lag)
    return df


def construir_feature_cols(df: pd.DataFrame) -> list:
    """Construye la lista completa de features incluyendo lags disponibles.

    Args:
        df: DataFrame con columnas base y de lag.

    Returns:
        Lista de nombres de columnas a usar como features.
    """
    lag_cols = [
        f'{col}_lag{lag}'
        for col in LAG_COLS_SOURCE
        for lag in LAG_PERIODS
        if f'{col}_lag{lag}' in df.columns
    ]
    return BASE_FEATURE_COLS + lag_cols


# ── Carga y split ──────────────────────────────────────────────────────────

def cargar_features(path: Path) -> tuple:
    """Carga el parquet, agrega lag features, filtra outliers y valida columnas.

    Args:
        path: Ruta a features.parquet.

    Returns:
        Tupla (df, feature_cols) lista para modelado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas.
    """
    if not path.exists():
        raise FileNotFoundError(
            f'No se encontró {path}. Ejecuta primero src/feature_engineering.py'
        )
    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    required = set(BASE_FEATURE_COLS) | {TARGET_CLF, TARGET_REG, 'Date', 'Symbol'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Columnas faltantes en features.parquet: {missing}')

    df = agregar_lag_features(df)
    feature_cols = construir_feature_cols(df)

    n_antes = len(df)
    df = df.dropna(subset=feature_cols + [TARGET_CLF, TARGET_REG])
    print(f'  Filas eliminadas por NaN (incluye lags): {n_antes - len(df)}')

    n_antes = len(df)
    df = df[df[TARGET_REG].between(RETURN_MIN, RETURN_MAX)].copy()
    print(f'  Filas eliminadas por outliers en Return: {n_antes - len(df)}')
    print(f'  Return limpio — media: {df[TARGET_REG].mean():.6f}  std: {df[TARGET_REG].std():.6f}')

    return df, feature_cols


def split_temporal(df: pd.DataFrame, val_size: float = 0.10,
                   test_size: float = 0.10):
    """Divide en train/val/test sin shuffle, respetando orden temporal.

    Args:
        df: DataFrame con columna Date.
        val_size: Fracción para validación.
        test_size: Fracción para test.

    Returns:
        Tupla (train, val, test).
    """
    fechas = df['Date'].sort_values().unique()
    n = len(fechas)
    corte_val  = fechas[int(n * (1 - val_size - test_size))]
    corte_test = fechas[int(n * (1 - test_size))]

    train = df[df['Date'] <  corte_val].copy()
    val   = df[(df['Date'] >= corte_val) & (df['Date'] < corte_test)].copy()
    test  = df[df['Date'] >= corte_test].copy()

    print(f'  Train: {len(train):>6} filas  ({train["Date"].min().date()} → {train["Date"].max().date()})')
    print(f'  Val  : {len(val):>6} filas  ({val["Date"].min().date()}  → {val["Date"].max().date()})')
    print(f'  Test : {len(test):>6} filas  ({test["Date"].min().date()}  → {test["Date"].max().date()})')
    return train, val, test


# ── Clasificadores ─────────────────────────────────────────────────────────

def construir_espacios_clf() -> dict:
    """Define pipelines base y espacios de búsqueda para clasificadores.

    Returns:
        Diccionario nombre → (pipeline_base, param_distributions).
    """
    return {
        'LogisticRegression': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
                )),
            ]),
            {'clf__C': [0.01, 0.1, 1.0, 10.0],
             'clf__solver': ['lbfgs', 'saga']},
        ),
        'RandomForestClassifier': (
            Pipeline([
                ('clf', RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
                )),
            ]),
            {'clf__n_estimators': [100, 200, 300],
             'clf__max_depth': [5, 10, 15, None],
             'clf__min_samples_leaf': [1, 3, 5]},
        ),
        'XGBClassifier': (
            Pipeline([
                ('clf', XGBClassifier(
                    random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0
                )),
            ]),
            {'clf__n_estimators': [100, 200, 300],
             'clf__max_depth': [3, 5, 7],
             'clf__learning_rate': [0.01, 0.05, 0.1],
             'clf__subsample': [0.7, 0.8, 1.0]},
        ),
    }


def entrenar_clasificadores(train, val, test, feature_cols):
    """Entrena los 3 clasificadores con RandomizedSearchCV y evalúa en val/test.

    Args:
        train: Datos de entrenamiento.
        val: Datos de validación.
        test: Datos de test.
        feature_cols: Lista de features a usar.

    Returns:
        Tupla (resultados_df, mejor_pipeline, mejor_nombre).
    """
    X_train = train[feature_cols].values
    y_train = train[TARGET_CLF].values
    X_val   = val[feature_cols].values
    y_val   = val[TARGET_CLF].values
    X_test  = test[feature_cols].values
    y_test  = test[TARGET_CLF].values

    tscv = TimeSeriesSplit(n_splits=5)
    espacios = construir_espacios_clf()
    resultados = []
    pipelines_ajustados = {}

    for nombre, (pipeline_base, param_dist) in espacios.items():
        print(f'    Tuning {nombre} ({N_ITER_SEARCH} iteraciones) ...')
        search = RandomizedSearchCV(
            pipeline_base, param_dist,
            n_iter=N_ITER_SEARCH, cv=tscv,
            scoring='roc_auc', n_jobs=-1,
            random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        pipelines_ajustados[nombre] = pipeline
        print(f'      Mejores params: {search.best_params_}')

        y_val_pred  = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)

        try:
            auc_val  = roc_auc_score(y_val,  pipeline.predict_proba(X_val)[:, 1])
            auc_test = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        except AttributeError:
            auc_val = auc_test = float('nan')

        resultados.append({
            'Modelo'         : nombre,
            'CV_AUC_mean'    : round(float(search.best_score_), 4),
            'Val_Acc'        : round(float(accuracy_score(y_val, y_val_pred)), 4),
            'Val_F1'         : round(float(f1_score(y_val, y_val_pred)), 4),
            'Val_AUC'        : round(float(auc_val), 4),
            'Test_Acc'       : round(float(accuracy_score(y_test, y_test_pred)), 4),
            'Test_F1'        : round(float(f1_score(y_test, y_test_pred)), 4),
            'Test_AUC'       : round(float(auc_test), 4),
            'Mejores_params' : str(search.best_params_),
        })

    resultados_df  = pd.DataFrame(resultados).set_index('Modelo')
    mejor_nombre   = resultados_df['Val_AUC'].idxmax()
    mejor_pipeline = pipelines_ajustados[mejor_nombre]
    print(f'\n  Mejor clasificador: {mejor_nombre}  (Val_AUC={resultados_df.loc[mejor_nombre, "Val_AUC"]})')
    return resultados_df, mejor_pipeline, mejor_nombre


# ── Regresores ─────────────────────────────────────────────────────────────

def construir_espacios_reg() -> dict:
    """Define pipelines base y espacios de búsqueda para regresores.

    Returns:
        Diccionario nombre → (pipeline_base, param_distributions).
    """
    return {
        'Ridge': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('reg', Ridge()),
            ]),
            {'reg__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        'RandomForestRegressor': (
            Pipeline([
                ('reg', RandomForestRegressor(
                    random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]),
            {'reg__n_estimators': [100, 200, 300],
             'reg__max_depth': [5, 10, 15, None],
             'reg__min_samples_leaf': [1, 3, 5]},
        ),
        'XGBRegressor': (
            Pipeline([
                ('reg', XGBRegressor(
                    random_state=RANDOM_STATE, verbosity=0
                )),
            ]),
            {'reg__n_estimators': [100, 200, 300],
             'reg__max_depth': [3, 5, 7],
             'reg__learning_rate': [0.01, 0.05, 0.1],
             'reg__subsample': [0.7, 0.8, 1.0]},
        ),
    }


def entrenar_regresores(train, val, test, feature_cols):
    """Entrena los 3 regresores con RandomizedSearchCV y evalúa en val/test.

    Target: retorno porcentual del día siguiente (Return).
    Se excluye Return y sus lags de las features para evitar data leakage.

    Args:
        train: Datos de entrenamiento.
        val: Datos de validación.
        test: Datos de test.
        feature_cols: Lista completa de features.

    Returns:
        Tupla (resultados_df, mejor_pipeline, mejor_nombre, feature_cols_reg).
    """
    feature_cols_reg = [
        c for c in feature_cols
        if c != TARGET_REG and not c.startswith('Return_lag')
    ]

    X_train = train[feature_cols_reg].values
    y_train = train[TARGET_REG].values
    X_val   = val[feature_cols_reg].values
    y_val   = val[TARGET_REG].values
    X_test  = test[feature_cols_reg].values
    y_test  = test[TARGET_REG].values

    tscv = TimeSeriesSplit(n_splits=5)
    espacios = construir_espacios_reg()
    resultados = []
    pipelines_ajustados = {}

    for nombre, (pipeline_base, param_dist) in espacios.items():
        print(f'    Tuning {nombre} ({N_ITER_SEARCH} iteraciones) ...')
        search = RandomizedSearchCV(
            pipeline_base, param_dist,
            n_iter=N_ITER_SEARCH, cv=tscv,
            scoring='r2', n_jobs=-1,
            random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        pipelines_ajustados[nombre] = pipeline
        print(f'      Mejores params: {search.best_params_}')

        y_val_pred  = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)

        resultados.append({
            'Modelo'         : nombre,
            'CV_R2_mean'     : round(float(search.best_score_), 4),
            'Val_MAE'        : round(float(mean_absolute_error(y_val, y_val_pred)), 6),
            'Val_RMSE'       : round(float(np.sqrt(mean_squared_error(y_val, y_val_pred))), 6),
            'Val_R2'         : round(float(r2_score(y_val, y_val_pred)), 4),
            'Test_MAE'       : round(float(mean_absolute_error(y_test, y_test_pred)), 6),
            'Test_RMSE'      : round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 6),
            'Test_R2'        : round(float(r2_score(y_test, y_test_pred)), 4),
            'Mejores_params' : str(search.best_params_),
        })

    resultados_df  = pd.DataFrame(resultados).set_index('Modelo')
    mejor_nombre   = resultados_df['Val_R2'].idxmax()
    mejor_pipeline = pipelines_ajustados[mejor_nombre]
    print(f'\n  Mejor regresor: {mejor_nombre}  (Val_R2={resultados_df.loc[mejor_nombre, "Val_R2"]})')
    return resultados_df, mejor_pipeline, mejor_nombre, feature_cols_reg


# ── Reporte ────────────────────────────────────────────────────────────────

def generar_reporte(res_clf, res_reg, mejor_clf_nombre, mejor_reg_nombre,
                    feature_cols, feature_cols_reg, n_train, n_val, n_test):
    """Genera reporte completo en Markdown con métricas y conclusiones.

    Args:
        res_clf: DataFrame con métricas de clasificadores.
        res_reg: DataFrame con métricas de regresores.
        mejor_clf_nombre: Nombre del mejor clasificador.
        mejor_reg_nombre: Nombre del mejor regresor.
        feature_cols: Features usadas en clasificación.
        feature_cols_reg: Features usadas en regresión.
        n_train: Número de filas de entrenamiento.
        n_val: Número de filas de validación.
        n_test: Número de filas de test.
    """
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M')
    lineas = [
        '# Reporte de Modelado Supervisado — Crypto ML Project',
        f'\n**Generado:** {fecha}',
        '\n---\n',
        '## 1. Configuración del experimento',
        f'- **Features base:** {len(BASE_FEATURE_COLS)}',
        f'- **Features con lags (t-1, t-3, t-7):** {len(feature_cols)}',
        f'- **Split temporal:** Train {n_train} filas | Val {n_val} filas | Test {n_test} filas',
        f'- **Validación cruzada:** TimeSeriesSplit (5 folds)',
        f'- **Tuning:** RandomizedSearchCV ({N_ITER_SEARCH} iteraciones)',
        f'- **Filtro outliers Return:** [{RETURN_MIN}, {RETURN_MAX}]',
        '\n---\n',
        '## 2. Clasificadores — predice si el precio sube o baja',
        '\n### Métricas comparativas\n',
        '| Modelo | CV_AUC | Val_Acc | Val_F1 | Val_AUC | Test_Acc | Test_F1 | Test_AUC |',
        '|--------|--------|---------|--------|---------|----------|---------|----------|',
    ]

    for nombre, row in res_clf.iterrows():
        lineas.append(
            f'| {nombre} | {row["CV_AUC_mean"]} | {row["Val_Acc"]} | '
            f'{row["Val_F1"]} | {row["Val_AUC"]} | {row["Test_Acc"]} | '
            f'{row["Test_F1"]} | {row["Test_AUC"]} |'
        )

    lineas += [
        f'\n### Mejor clasificador: **{mejor_clf_nombre}**',
        f'- Val AUC: {res_clf.loc[mejor_clf_nombre, "Val_AUC"]}',
        f'- Test AUC: {res_clf.loc[mejor_clf_nombre, "Test_AUC"]}',
        f'- Hiperparámetros: `{res_clf.loc[mejor_clf_nombre, "Mejores_params"]}`',
        '\n### Interpretación',
        '- AUC > 0.5 indica que los modelos capturan alguna señal real en los datos.',
        '- Predecir movimientos de criptomonedas es intrínsecamente difícil por la alta volatilidad.',
        '- Los features de lag aportan memoria temporal que mejora la capacidad predictiva.',
        '\n---\n',
        '## 3. Regresores — predice el retorno porcentual del día siguiente',
        '\n### Métricas comparativas\n',
        '| Modelo | CV_R2 | Val_MAE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 |',
        '|--------|-------|---------|----------|--------|----------|-----------|---------|',
    ]

    for nombre, row in res_reg.iterrows():
        lineas.append(
            f'| {nombre} | {row["CV_R2_mean"]} | {row["Val_MAE"]} | '
            f'{row["Val_RMSE"]} | {row["Val_R2"]} | {row["Test_MAE"]} | '
            f'{row["Test_RMSE"]} | {row["Test_R2"]} |'
        )

    lineas += [
        f'\n### Mejor regresor: **{mejor_reg_nombre}**',
        f'- Val R²: {res_reg.loc[mejor_reg_nombre, "Val_R2"]}',
        f'- Test R²: {res_reg.loc[mejor_reg_nombre, "Test_R2"]}',
        f'- Hiperparámetros: `{res_reg.loc[mejor_reg_nombre, "Mejores_params"]}`',
        '\n### Interpretación',
        '- El target es el retorno porcentual diario — variable estacionaria con sentido financiero.',
        '- R² positivo en test indica que el modelo generaliza más allá del azar.',
        '- Return fue excluido de las features del regresor para evitar data leakage.',
        '- Los lags de rsi_14 y macd (t-1, t-3, t-7) capturan inercia de los indicadores técnicos.',
        '\n---\n',
        '## 4. Figuras generadas',
        '- `13_comparacion_clasificadores.png` — AUC val vs test por modelo',
        '- `14_comparacion_regresores.png` — R² val vs test por modelo',
        '- `15_confusion_matrix.png` — Matriz de confusión del mejor clasificador',
        '- `16_roc_curve.png` — Curva ROC del mejor clasificador',
        '- `17_feature_importance_clf.png` — Importancia de features (clasificador)',
        '- `18_feature_importance_reg.png` — Importancia de features (regresor)',
        '- `19_predicciones_vs_real.png` — Predicciones vs valores reales (regresor)',
        '- `20_residuos.png` — Distribución de residuos (regresor)',
    ]

    reporte_path = REPORTS_DIR / 'supervised_report.md'
    reporte_path.write_text('\n'.join(lineas), encoding='utf-8')
    print('[ok] supervised_report.md')


# ── Figuras ────────────────────────────────────────────────────────────────

def plot_comparacion_clasificadores(resultados_df):
    """Barras comparando AUC en val y test para los 3 clasificadores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(resultados_df))
    w = 0.35
    ax.bar(x - w / 2, resultados_df['Val_AUC'],  w, label='Val AUC',  color='steelblue')
    ax.bar(x + w / 2, resultados_df['Test_AUC'], w, label='Test AUC', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(resultados_df.index, rotation=15)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Comparación de clasificadores — AUC Val vs Test')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '13_comparacion_clasificadores.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 13_comparacion_clasificadores.png')


def plot_comparacion_regresores(resultados_df):
    """Barras comparando R² en val y test para los 3 regresores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(resultados_df))
    w = 0.35
    ax.bar(x - w / 2, resultados_df['Val_R2'],  w, label='Val R²',  color='mediumseagreen')
    ax.bar(x + w / 2, resultados_df['Test_R2'], w, label='Test R²', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(resultados_df.index, rotation=15)
    ax.set_ylabel('R²')
    ax.set_title('Comparación de regresores — R² Val vs Test (target: Return)')
    ax.legend()
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '14_comparacion_regresores.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 14_comparacion_regresores.png')


def plot_confusion_matrix(pipeline, X_test, y_test, nombre):
    """Matriz de confusión del mejor clasificador."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Baja (0)', 'Sube (1)'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matriz de Confusión — {nombre}')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '15_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 15_confusion_matrix.png')


def plot_roc_curve(pipeline, X_test, y_test, nombre):
    """Curva ROC del mejor clasificador."""
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        print(f'  [{nombre}] no tiene predict_proba — curva ROC omitida.')
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=nombre)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title(f'Curva ROC — {nombre}')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '16_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 16_roc_curve.png')


def plot_feature_importance(pipeline, feature_cols, nombre, fig_num):
    """Importancia de features (top 20) si el modelo lo soporta."""
    estimador = pipeline.named_steps.get('clf') or pipeline.named_steps.get('reg')
    if not hasattr(estimador, 'feature_importances_'):
        print(f'  [{nombre}] no tiene feature_importances_ — figura omitida.')
        return
    importances = pd.Series(estimador.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Importancia de Features — {nombre} (top 20)')
    ax.set_xlabel('Importancia')
    plt.tight_layout()
    fname = f'{fig_num}_feature_importance_{nombre.lower()[:3]}.png'
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] {fname}')


def plot_predicciones_vs_real(pipeline, X_test, y_test, nombre):
    """Scatter de predicciones vs valores reales del mejor regresor."""
    y_pred = pipeline.predict(X_test)
    idx = np.random.RandomState(42).choice(len(y_test), min(500, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test[idx], y_pred[idx], alpha=0.4, s=15, color='steelblue')
    lim = max(abs(y_test[idx]).max(), abs(y_pred[idx]).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='Predicción perfecta')
    ax.set_xlabel('Return real')
    ax.set_ylabel('Return predicho')
    ax.set_title(f'Predicciones vs Real — {nombre}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '19_predicciones_vs_real.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 19_predicciones_vs_real.png')


def plot_residuos(pipeline, X_test, y_test, nombre):
    """Histograma y scatter de residuos del mejor regresor."""
    y_pred = pipeline.predict(X_test)
    residuos = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuos, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Residuo (real − predicho)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title(f'Distribución de Residuos — {nombre}')

    axes[1].scatter(y_pred, residuos, alpha=0.3, s=10, color='darkorange')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Return predicho')
    axes[1].set_ylabel('Residuo')
    axes[1].set_title('Residuos vs Predicciones')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '20_residuos.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 20_residuos.png')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    """Ejecuta el pipeline completo de modelado supervisado."""
    print('[supervisado] Cargando features y agregando lags ...')
    df, feature_cols = cargar_features(DATA_PATH)
    print(f'  Shape: {df.shape}  |  Monedas: {df["Symbol"].nunique()}')
    print(f'  Total features (base + lags): {len(feature_cols)}')
    print(f'  Balance target clf: {df[TARGET_CLF].value_counts().to_dict()}')

    print('\n[supervisado] Split temporal 80/10/10 ...')
    train, val, test = split_temporal(df)

    # ── Clasificadores ─────────────────────────────────────────────────────
    print('\n[supervisado] Tuning y entrenamiento de clasificadores ...')
    res_clf, mejor_clf, mejor_clf_nombre = entrenar_clasificadores(
        train, val, test, feature_cols
    )
    print('\n  Tabla comparativa — clasificadores:')
    print(res_clf[['CV_AUC_mean', 'Val_Acc', 'Val_F1', 'Val_AUC',
                   'Test_Acc', 'Test_F1', 'Test_AUC']].to_string())

    print(f'\n[supervisado] Guardando mejor clasificador ({mejor_clf_nombre}) ...')
    joblib.dump(mejor_clf, MODELS_DIR / 'best_classifier.pkl')
    print('  → models/best_classifier.pkl')

    # ── Regresores ─────────────────────────────────────────────────────────
    print('\n[supervisado] Tuning y entrenamiento de regresores (target: Return) ...')
    res_reg, mejor_reg, mejor_reg_nombre, feature_cols_reg = entrenar_regresores(
        train, val, test, feature_cols
    )
    print('\n  Tabla comparativa — regresores:')
    print(res_reg[['CV_R2_mean', 'Val_MAE', 'Val_RMSE', 'Val_R2',
                   'Test_MAE', 'Test_RMSE', 'Test_R2']].to_string())

    print(f'\n[supervisado] Guardando mejor regresor ({mejor_reg_nombre}) ...')
    joblib.dump(mejor_reg, MODELS_DIR / 'best_regressor.pkl')
    print('  → models/best_regressor.pkl')

    # ── Figuras ────────────────────────────────────────────────────────────
    print('\n[supervisado] Generando figuras ...')
    plot_comparacion_clasificadores(res_clf)
    plot_comparacion_regresores(res_reg)

    X_test_clf = test[feature_cols].values
    y_test_clf = test[TARGET_CLF].values
    plot_confusion_matrix(mejor_clf, X_test_clf, y_test_clf, mejor_clf_nombre)
    plot_roc_curve(mejor_clf, X_test_clf, y_test_clf, mejor_clf_nombre)
    plot_feature_importance(mejor_clf, feature_cols, mejor_clf_nombre, 17)

    X_test_reg = test[feature_cols_reg].values
    y_test_reg = test[TARGET_REG].values
    plot_feature_importance(mejor_reg, feature_cols_reg, mejor_reg_nombre, 18)
    plot_predicciones_vs_real(mejor_reg, X_test_reg, y_test_reg, mejor_reg_nombre)
    plot_residuos(mejor_reg, X_test_reg, y_test_reg, mejor_reg_nombre)

    # ── Reporte ────────────────────────────────────────────────────────────
    print('\n[supervisado] Generando reporte ...')
    generar_reporte(
        res_clf, res_reg, mejor_clf_nombre, mejor_reg_nombre,
        feature_cols, feature_cols_reg,
        len(train), len(val), len(test)
    )

    print('\n[done] Pipeline supervisado completado.')


if __name__ == '__main__':
    main()
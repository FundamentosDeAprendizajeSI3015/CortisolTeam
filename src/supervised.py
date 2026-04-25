"""
[04] Supervisado — Crypto ML Project
Entrena clasificadores y un regresor sobre features técnicos de criptomonedas.

Uso:
    python src/supervised.py

Salidas:
    models/best_classifier.pkl
    models/best_regressor.pkl
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data' / 'processed' / 'features.parquet'
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'reports' / 'figures'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'sma_7', 'sma_14', 'sma_30',
    'ema_14', 'rsi_14',
    'bb_high', 'bb_low', 'bb_width',
    'macd', 'macd_signal',
    'Return',
]
TARGET_CLF = 'target'
TARGET_REG = 'Close'

RANDOM_STATE = 42


# ── Carga y split ──────────────────────────────────────────────────────────

def cargar_features(path: Path) -> pd.DataFrame:
    """Carga el parquet de features y valida columnas requeridas.

    Args:
        path: Ruta a features.parquet.

    Returns:
        DataFrame listo para modelado.

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

    required = set(FEATURE_COLS) | {TARGET_CLF, TARGET_REG, 'Date', 'Symbol'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Columnas faltantes en features.parquet: {missing}')

    n_antes = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_CLF, TARGET_REG])
    print(f'  Filas eliminadas por NaN en features/target: {n_antes - len(df)}')
    return df


def split_temporal(df: pd.DataFrame, val_size: float = 0.10,
                   test_size: float = 0.10):
    """Divide el dataset en train/val/test sin shuffle, respetando el orden temporal.

    El corte se hace sobre fechas globales para mantener la integridad temporal
    entre todos los símbolos.

    Args:
        df: DataFrame con columna Date.
        val_size: Fracción destinada a validación (por defecto 0.10).
        test_size: Fracción destinada a test (por defecto 0.10).

    Returns:
        Tupla (train, val, test) de DataFrames.
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


# ── Entrenamiento de clasificadores ───────────────────────────────────────

def construir_clasificadores() -> dict:
    """Instancia los pipelines de clasificación con preprocesado incluido.

    Returns:
        Diccionario nombre → Pipeline de sklearn.
    """
    return {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
            )),
        ]),
        'RandomForestClassifier': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
            )),
        ]),
        'XGBClassifier': Pipeline([
            ('clf', XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, eval_metric='logloss',
                verbosity=0,
            )),
        ]),
    }


def entrenar_clasificadores(train: pd.DataFrame, val: pd.DataFrame,
                             test: pd.DataFrame) -> tuple:
    """Entrena clasificadores con TimeSeriesSplit y evalúa en val y test.

    Args:
        train: Datos de entrenamiento.
        val: Datos de validación.
        test: Datos de test.

    Returns:
        Tupla (resultados_df, mejor_pipeline) donde resultados_df contiene
        métricas comparativas y mejor_pipeline es el mejor modelo ajustado.
    """
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_CLF].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_CLF].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_CLF].values

    tscv = TimeSeriesSplit(n_splits=5)
    modelos = construir_clasificadores()
    resultados = []

    for nombre, pipeline in modelos.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=tscv, scoring='roc_auc', n_jobs=-1)
        pipeline.fit(X_train, y_train)

        y_val_pred  = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)

        try:
            y_val_proba  = pipeline.predict_proba(X_val)[:, 1]
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_val  = roc_auc_score(y_val, y_val_proba)
            auc_test = roc_auc_score(y_test, y_test_proba)
        except AttributeError:
            auc_val = auc_test = float('nan')

        resultados.append({
            'Modelo'      : nombre,
            'CV_AUC_mean' : round(float(cv_scores.mean()), 4),
            'CV_AUC_std'  : round(float(cv_scores.std()), 4),
            'Val_Acc'     : round(float(accuracy_score(y_val, y_val_pred)), 4),
            'Val_F1'      : round(float(f1_score(y_val, y_val_pred)), 4),
            'Val_AUC'     : round(float(auc_val), 4),
            'Test_Acc'    : round(float(accuracy_score(y_test, y_test_pred)), 4),
            'Test_F1'     : round(float(f1_score(y_test, y_test_pred)), 4),
            'Test_AUC'    : round(float(auc_test), 4),
        })

    resultados_df = pd.DataFrame(resultados).set_index('Modelo')

    mejor_nombre = resultados_df['Val_AUC'].idxmax()
    mejor_pipeline = modelos[mejor_nombre]
    print(f'\n  Mejor clasificador: {mejor_nombre}  (Val_AUC={resultados_df.loc[mejor_nombre, "Val_AUC"]})')
    return resultados_df, mejor_pipeline, mejor_nombre


# ── Entrenamiento del regresor ────────────────────────────────────────────

def entrenar_regresor(train: pd.DataFrame, val: pd.DataFrame,
                      test: pd.DataFrame) -> tuple:
    """Entrena RandomForestRegressor para predecir el precio de cierre del día siguiente.

    Args:
        train: Datos de entrenamiento.
        val: Datos de validación.
        test: Datos de test.

    Returns:
        Tupla (métricas_dict, pipeline_ajustado).
    """
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_REG].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_REG].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_REG].values

    pipeline = Pipeline([
        ('reg', RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=tscv, scoring='r2', n_jobs=-1)
    pipeline.fit(X_train, y_train)

    y_val_pred  = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    metricas = {
        'CV_R2_mean' : round(float(cv_scores.mean()), 4),
        'CV_R2_std'  : round(float(cv_scores.std()), 4),
        'Val_MAE'    : round(float(mean_absolute_error(y_val, y_val_pred)), 4),
        'Val_RMSE'   : round(float(np.sqrt(mean_squared_error(y_val, y_val_pred))), 4),
        'Val_R2'     : round(float(r2_score(y_val, y_val_pred)), 4),
        'Test_MAE'   : round(float(mean_absolute_error(y_test, y_test_pred)), 4),
        'Test_RMSE'  : round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
        'Test_R2'    : round(float(r2_score(y_test, y_test_pred)), 4),
    }
    return metricas, pipeline


# ── Figuras ────────────────────────────────────────────────────────────────

def plot_comparacion_modelos(resultados_df: pd.DataFrame) -> None:
    """Genera gráfico de barras comparando AUC en validación y test.

    Args:
        resultados_df: DataFrame con métricas por modelo.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(resultados_df))
    width = 0.35
    ax.bar(x - width / 2, resultados_df['Val_AUC'],  width, label='Val AUC',  color='steelblue')
    ax.bar(x + width / 2, resultados_df['Test_AUC'], width, label='Test AUC', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(resultados_df.index, rotation=15)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Comparación de clasificadores — AUC en Val y Test')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '13_comparacion_modelos.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 13_comparacion_modelos.png')


def plot_confusion_matrix(pipeline, X_test: np.ndarray,
                           y_test: np.ndarray, nombre: str) -> None:
    """Guarda la matriz de confusión del mejor clasificador.

    Args:
        pipeline: Pipeline ajustado.
        X_test: Features de test.
        y_test: Etiquetas reales de test.
        nombre: Nombre del modelo para el título.
    """
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Baja (0)', 'Sube (1)'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matriz de Confusión — {nombre}')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '14_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 14_confusion_matrix.png')


def plot_roc_curve(pipeline, X_test: np.ndarray,
                   y_test: np.ndarray, nombre: str) -> None:
    """Guarda la curva ROC del mejor clasificador.

    Args:
        pipeline: Pipeline ajustado con predict_proba.
        X_test: Features de test.
        y_test: Etiquetas reales de test.
        nombre: Nombre del modelo para el título.
    """
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
    fig.savefig(FIGURES_DIR / '15_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 15_roc_curve.png')


def plot_feature_importance(pipeline, nombre: str) -> None:
    """Guarda gráfico de importancia de features si el modelo lo soporta.

    Args:
        pipeline: Pipeline ajustado.
        nombre: Nombre del modelo.
    """
    estimador = pipeline.named_steps.get('clf') or pipeline.named_steps.get('reg')
    if not hasattr(estimador, 'feature_importances_'):
        return

    importances = pd.Series(estimador.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Importancia de Features — {nombre}')
    ax.set_xlabel('Importancia')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '16_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 16_feature_importance.png')


def main():
    """Ejecuta el pipeline completo de modelado supervisado."""
    print('[supervisado] Cargando features ...')
    df = cargar_features(DATA_PATH)
    print(f'  Shape: {df.shape}  |  Monedas: {df["Symbol"].nunique()}')
    print(f'  Balance target: {df[TARGET_CLF].value_counts().to_dict()}')

    print('\n[supervisado] Split temporal 80/10/10 ...')
    train, val, test = split_temporal(df)

    # ── Clasificadores ─────────────────────────────────────────────────────
    print('\n[supervisado] Entrenando clasificadores ...')
    resultados_df, mejor_clf, mejor_clf_nombre = entrenar_clasificadores(train, val, test)

    print('\n  Tabla comparativa de clasificadores:')
    print(resultados_df.to_string())

    print(f'\n[supervisado] Guardando mejor clasificador ({mejor_clf_nombre}) ...')
    joblib.dump(mejor_clf, MODELS_DIR / 'best_classifier.pkl')
    print(f'  → models/best_classifier.pkl')

    # ── Regresor ───────────────────────────────────────────────────────────
    print('\n[supervisado] Entrenando regresor (RandomForestRegressor) ...')
    metricas_reg, mejor_reg = entrenar_regresor(train, val, test)

    print('\n  Métricas del regresor:')
    for k, v in metricas_reg.items():
        print(f'    {k:<15}: {v}')

    print('\n[supervisado] Guardando regresor ...')
    joblib.dump(mejor_reg, MODELS_DIR / 'best_regressor.pkl')
    print('  → models/best_regressor.pkl')

    # ── Figuras ────────────────────────────────────────────────────────────
    print('\n[supervisado] Generando figuras ...')
    plot_comparacion_modelos(resultados_df)

    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_CLF].values
    plot_confusion_matrix(mejor_clf, X_test, y_test, mejor_clf_nombre)
    plot_roc_curve(mejor_clf, X_test, y_test, mejor_clf_nombre)
    plot_feature_importance(mejor_clf, mejor_clf_nombre)

    print('\n[done] Pipeline supervisado completado.')


if __name__ == '__main__':
    main()

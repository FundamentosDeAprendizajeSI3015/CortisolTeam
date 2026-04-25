"""
[06] Scoring — Crypto ML Project
Evaluates all supervised and unsupervised models with standard metrics,
generates a consolidated comparison table, runs a simple backtesting
simulation, and produces the final PDF report.

Usage:
    python src/scoring.py

Outputs:
    reports/metrics_summary.csv   — consolidated metrics for every model
    reports/model_report.pdf      — final PDF report with tables and figures
    reports/figures/17_*           — scoring-specific figures
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
from datetime import datetime

# ── Sklearn metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from xgboost import XGBClassifier

# ── PDF generation ──────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / 'data' / 'processed' / 'features.parquet'
CLUST_FEAT   = ROOT / 'data' / 'features_clustering.csv'
CLUST_LABELS = ROOT / 'data' / 'cluster_labels.csv'
RAW_DATA     = ROOT / 'data' / 'crypto_raw.csv'
MODELS_DIR   = ROOT / 'models'
REPORTS_DIR  = ROOT / 'reports'
FIGURES_DIR  = REPORTS_DIR / 'figures'
PDF_PATH     = REPORTS_DIR / 'model_report.pdf'
CSV_PATH     = REPORTS_DIR / 'metrics_summary.csv'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns (same as supervised.py) ─────────────────────────────────
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

# ── Clustering parameters (same as clustering.py) ──────────────────────────
K_BEST = 2
K_ALT  = 4


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

def load_features(path: Path) -> pd.DataFrame:
    """Load the features parquet file and validate required columns.

    Args:
        path: Path to features.parquet.

    Returns:
        Clean DataFrame ready for modelling.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run src/feature_engineering.py first.'
        )
    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])

    required = set(FEATURE_COLS) | {TARGET_CLF, TARGET_REG, 'Date', 'Symbol'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in features.parquet: {missing}')

    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_CLF, TARGET_REG])
    print(f'  Rows dropped due to NaN: {n_before - len(df)}')
    return df


def split_temporal(df: pd.DataFrame, val_size: float = 0.10,
                   test_size: float = 0.10):
    """Split dataset chronologically into train / val / test sets.

    The split is performed on global dates so that temporal integrity
    is maintained across all symbols.

    Args:
        df: DataFrame with a Date column.
        val_size: Fraction allocated to validation (default 0.10).
        test_size: Fraction allocated to test (default 0.10).

    Returns:
        Tuple (train, val, test) DataFrames.
    """
    dates = df['Date'].sort_values().unique()
    n = len(dates)
    cut_val  = dates[int(n * (1 - val_size - test_size))]
    cut_test = dates[int(n * (1 - test_size))]

    train = df[df['Date'] <  cut_val].copy()
    val   = df[(df['Date'] >= cut_val) & (df['Date'] < cut_test)].copy()
    test  = df[df['Date'] >= cut_test].copy()

    print(f'  Train : {len(train):>6} rows  '
          f'({train["Date"].min().date()} → {train["Date"].max().date()})')
    print(f'  Val   : {len(val):>6} rows  '
          f'({val["Date"].min().date()} → {val["Date"].max().date()})')
    print(f'  Test  : {len(test):>6} rows  '
          f'({test["Date"].min().date()} → {test["Date"].max().date()})')
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD & TRAIN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════

def build_classifiers() -> dict:
    """Instantiate classification pipelines with embedded pre-processing.

    Returns:
        Dict mapping model name → sklearn Pipeline.
    """
    return {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                class_weight='balanced',
            )),
        ]),
        'RandomForestClassifier': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
                class_weight='balanced',
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


def train_classifiers(train, val, test):
    """Train every classifier, compute extended metrics on val and test sets.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.

    Returns:
        Tuple (results_df, best_pipeline, best_name) with comparative metrics.
    """
    X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET_CLF].values
    X_va, y_va = val[FEATURE_COLS].values,   val[TARGET_CLF].values
    X_te, y_te = test[FEATURE_COLS].values,  test[TARGET_CLF].values

    tscv = TimeSeriesSplit(n_splits=5)
    models = build_classifiers()
    results = []

    for name, pipe in models.items():
        # Cross-validation on training data
        cv_scores = cross_val_score(
            pipe, X_tr, y_tr, cv=tscv, scoring='roc_auc', n_jobs=-1
        )
        pipe.fit(X_tr, y_tr)

        y_va_pred = pipe.predict(X_va)
        y_te_pred = pipe.predict(X_te)

        # Probabilities for AUC
        try:
            y_va_proba = pipe.predict_proba(X_va)[:, 1]
            y_te_proba = pipe.predict_proba(X_te)[:, 1]
            auc_va = roc_auc_score(y_va, y_va_proba)
            auc_te = roc_auc_score(y_te, y_te_proba)
        except AttributeError:
            auc_va = auc_te = float('nan')

        results.append({
            'Model':          name,
            'Type':           'Classification',
            'CV_AUC_mean':    round(float(cv_scores.mean()), 4),
            'CV_AUC_std':     round(float(cv_scores.std()), 4),
            'Val_Accuracy':   round(float(accuracy_score(y_va, y_va_pred)), 4),
            'Val_Precision':  round(float(precision_score(y_va, y_va_pred)), 4),
            'Val_Recall':     round(float(recall_score(y_va, y_va_pred)), 4),
            'Val_F1':         round(float(f1_score(y_va, y_va_pred)), 4),
            'Val_ROC_AUC':    round(float(auc_va), 4),
            'Test_Accuracy':  round(float(accuracy_score(y_te, y_te_pred)), 4),
            'Test_Precision': round(float(precision_score(y_te, y_te_pred)), 4),
            'Test_Recall':    round(float(recall_score(y_te, y_te_pred)), 4),
            'Test_F1':        round(float(f1_score(y_te, y_te_pred)), 4),
            'Test_ROC_AUC':   round(float(auc_te), 4),
        })

    results_df = pd.DataFrame(results).set_index('Model')
    best_name = results_df['Val_ROC_AUC'].idxmax()
    best_pipe = models[best_name]
    print(f'\n  Best classifier: {best_name}  '
          f'(Val_AUC={results_df.loc[best_name, "Val_ROC_AUC"]})')
    return results_df, best_pipe, best_name


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD & TRAIN REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════

def train_regressor(train, val, test):
    """Train RandomForestRegressor for next-day close price prediction.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.

    Returns:
        Tuple (metrics_dict, fitted_pipeline).
    """
    X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET_REG].values
    X_va, y_va = val[FEATURE_COLS].values,   val[TARGET_REG].values
    X_te, y_te = test[FEATURE_COLS].values,  test[TARGET_REG].values

    pipe = Pipeline([
        ('reg', RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=tscv,
                                scoring='r2', n_jobs=-1)
    pipe.fit(X_tr, y_tr)

    y_va_pred = pipe.predict(X_va)
    y_te_pred = pipe.predict(X_te)

    # MAPE — avoid division by zero
    def mape(y_true, y_pred):
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    metrics = {
        'Model':       'RandomForestRegressor',
        'Type':        'Regression',
        'CV_R2_mean':  round(float(cv_scores.mean()), 4),
        'CV_R2_std':   round(float(cv_scores.std()), 4),
        'Val_MAE':     round(float(mean_absolute_error(y_va, y_va_pred)), 4),
        'Val_RMSE':    round(float(np.sqrt(mean_squared_error(y_va, y_va_pred))), 4),
        'Val_MAPE':    round(mape(y_va, y_va_pred), 4),
        'Val_R2':      round(float(r2_score(y_va, y_va_pred)), 4),
        'Test_MAE':    round(float(mean_absolute_error(y_te, y_te_pred)), 4),
        'Test_RMSE':   round(float(np.sqrt(mean_squared_error(y_te, y_te_pred))), 4),
        'Test_MAPE':   round(mape(y_te, y_te_pred), 4),
        'Test_R2':     round(float(r2_score(y_te, y_te_pred)), 4),
    }
    return metrics, pipe


# ═══════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_clustering():
    """Recompute clustering algorithms and evaluate with internal metrics.

    Uses the same feature matrix and algorithms as the unsupervised branch
    (K-Means, DBSCAN, Agglomerative) and computes Silhouette Score,
    Davies-Bouldin Index, and Calinski-Harabasz Index.

    Returns:
        DataFrame with clustering metrics per algorithm.
    """
    if not CLUST_FEAT.exists():
        print(f'  [warn] {CLUST_FEAT} not found — skipping clustering eval.')
        return pd.DataFrame()

    features = pd.read_csv(CLUST_FEAT, index_col='Symbol')
    X = features.values

    algorithms = {
        f'KMeans_K{K_BEST}': KMeans(
            n_clusters=K_BEST, random_state=42, n_init=10
        ),
        f'KMeans_K{K_ALT}': KMeans(
            n_clusters=K_ALT, random_state=42, n_init=10
        ),
        'DBSCAN': DBSCAN(eps=1.5, min_samples=2),
        f'Agglomerative_K{K_BEST}': AgglomerativeClustering(
            n_clusters=K_BEST
        ),
    }

    results = []
    for name, algo in algorithms.items():
        labels = algo.fit_predict(X)
        n_clusters = len(set(labels) - {-1})

        # Skip metrics if only one cluster or all noise
        if n_clusters < 2:
            results.append({
                'Model': name, 'Type': 'Clustering',
                'N_Clusters': n_clusters,
                'Silhouette': float('nan'),
                'Davies_Bouldin': float('nan'),
                'Calinski_Harabasz': float('nan'),
            })
            continue

        # For DBSCAN, mask noise points for metric computation
        mask = labels != -1
        if mask.sum() < 2 or len(set(labels[mask])) < 2:
            results.append({
                'Model': name, 'Type': 'Clustering',
                'N_Clusters': n_clusters,
                'Silhouette': float('nan'),
                'Davies_Bouldin': float('nan'),
                'Calinski_Harabasz': float('nan'),
            })
            continue

        X_eval = X[mask] if name == 'DBSCAN' else X
        lab_eval = labels[mask] if name == 'DBSCAN' else labels

        results.append({
            'Model':             name,
            'Type':              'Clustering',
            'N_Clusters':        n_clusters,
            'Silhouette':        round(float(silhouette_score(X_eval, lab_eval)), 4),
            'Davies_Bouldin':    round(float(davies_bouldin_score(X_eval, lab_eval)), 4),
            'Calinski_Harabasz': round(float(calinski_harabasz_score(X_eval, lab_eval)), 4),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 5. BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

def backtest(test: pd.DataFrame, classifier_pipe, symbol: str = 'BTC'):
    """Run a simple backtesting simulation: model-based buy/sell vs buy & hold.

    Strategy:
        - Each day, the classifier predicts whether the price will go up.
        - If prediction == 1 → hold/buy; if 0 → sell/stay out.
        - Compare cumulative returns against buying and holding.

    Args:
        test: Test DataFrame with Date, Symbol, Close, feature columns.
        classifier_pipe: Fitted classification pipeline.
        symbol: Cryptocurrency symbol to simulate on (default 'BTC').

    Returns:
        Tuple (backtest_df, summary_dict) with daily positions and returns.
    """
    coin = test[test['Symbol'] == symbol].copy().sort_values('Date')

    if len(coin) < 2:
        print(f'  [warn] Not enough data for {symbol} in test set.')
        return pd.DataFrame(), {}

    X_coin = coin[FEATURE_COLS].values
    preds = classifier_pipe.predict(X_coin)

    coin = coin.reset_index(drop=True)
    coin['Prediction'] = preds
    coin['Daily_Return'] = coin['Close'].pct_change()

    # Strategy return: hold when prediction == 1, otherwise 0
    coin['Strategy_Return'] = coin['Daily_Return'] * coin['Prediction']

    # Cumulative returns
    coin['BuyHold_Cum'] = (1 + coin['Daily_Return']).cumprod()
    coin['Strategy_Cum'] = (1 + coin['Strategy_Return']).cumprod()

    # Fill first NaN
    coin['BuyHold_Cum'] = coin['BuyHold_Cum'].fillna(1.0)
    coin['Strategy_Cum'] = coin['Strategy_Cum'].fillna(1.0)

    # Summary statistics
    final_bh = coin['BuyHold_Cum'].iloc[-1]
    final_st = coin['Strategy_Cum'].iloc[-1]
    total_days = len(coin)
    days_in = int(coin['Prediction'].sum())

    summary = {
        'Symbol':                   symbol,
        'Test_Days':                total_days,
        'Days_In_Market':           days_in,
        'Days_Out_Market':          total_days - days_in,
        'BuyHold_Total_Return':     round(float((final_bh - 1) * 100), 2),
        'Strategy_Total_Return':    round(float((final_st - 1) * 100), 2),
        'Strategy_vs_BuyHold_pct':  round(float((final_st / final_bh - 1) * 100), 2),
    }
    return coin, summary


# ═══════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_classification_comparison(clf_df: pd.DataFrame):
    """Bar chart comparing classification metrics across models.

    Args:
        clf_df: DataFrame with classification metrics indexed by model name.
    """
    metrics_to_plot = ['Test_Accuracy', 'Test_Precision', 'Test_Recall',
                       'Test_F1', 'Test_ROC_AUC']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(clf_df))
    width = 0.15
    colors_list = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    for i, metric in enumerate(metrics_to_plot):
        if metric in clf_df.columns:
            ax.bar(x + i * width, clf_df[metric], width,
                   label=metric.replace('Test_', ''), color=colors_list[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(clf_df.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Classification Models — Test Set Metrics Comparison')
    ax.legend(loc='lower right')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '17_scoring_clf_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 17_scoring_clf_comparison.png')


def plot_confusion_matrices(models_dict, X_test, y_test):
    """Plot confusion matrix for every classifier side by side.

    Args:
        models_dict: Dict mapping name → fitted pipeline.
        X_test: Test feature array.
        y_test: Test labels array.
    """
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, pipe) in zip(axes, models_dict.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Down (0)', 'Up (1)'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}')

    plt.suptitle('Confusion Matrices — Test Set', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '18_scoring_confusion_matrices.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 18_scoring_confusion_matrices.png')


def plot_roc_curves(models_dict, X_test, y_test):
    """Overlay ROC curves for all classifiers on a single plot.

    Args:
        models_dict: Dict mapping name → fitted pipeline.
        X_test: Test feature array.
        y_test: Test labels array.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ['#4e79a7', '#f28e2b', '#e15759']

    for (name, pipe), color in zip(models_dict.items(), palette):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(
                y_test, y_proba, ax=ax, name=name, color=color
            )
        except AttributeError:
            pass

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title('ROC Curves — All Classifiers (Test Set)')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '19_scoring_roc_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 19_scoring_roc_curves.png')


def plot_regression_scatter(pipe, X_test, y_test):
    """Scatter plot of predicted vs actual close prices.

    Args:
        pipe: Fitted regression pipeline.
        X_test: Test feature array.
        y_test: Actual close prices.
    """
    y_pred = pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#4e79a7')

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5,
            label='Perfect Prediction')

    ax.set_xlabel('Actual Close Price')
    ax.set_ylabel('Predicted Close Price')
    ax.set_title('Regression — Predicted vs Actual (Test Set)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '20_scoring_regression_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 20_scoring_regression_scatter.png')


def plot_regression_residuals(pipe, X_test, y_test):
    """Residual distribution plot for the regressor.

    Args:
        pipe: Fitted regression pipeline.
        X_test: Test feature array.
        y_test: Actual close prices.
    """
    y_pred = pipe.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='#e15759')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('Predicted Close Price')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted')

    # Histogram of residuals
    axes[1].hist(residuals, bins=50, color='#76b7b2', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.suptitle('Regression Residuals Analysis', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '21_scoring_residuals.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 21_scoring_residuals.png')


def plot_clustering_metrics(clust_df: pd.DataFrame):
    """Bar chart comparing clustering quality metrics.

    Args:
        clust_df: DataFrame with clustering metrics.
    """
    if clust_df.empty:
        return

    plot_df = clust_df.dropna(subset=['Silhouette']).set_index('Model')
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_list = ['#59a14f', '#e15759', '#4e79a7']
    metrics = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
    titles = ['Silhouette Score (higher = better)',
              'Davies-Bouldin Index (lower = better)',
              'Calinski-Harabasz Index (higher = better)']

    for ax, metric, title, c in zip(axes, metrics, titles, colors_list):
        if metric in plot_df.columns:
            plot_df[metric].plot(kind='bar', ax=ax, color=c, edgecolor='white')
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(metric.replace('_', ' '))
            ax.tick_params(axis='x', rotation=25)

    plt.suptitle('Clustering Quality Metrics', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '22_scoring_clustering_metrics.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 22_scoring_clustering_metrics.png')


def plot_backtest(bt_df: pd.DataFrame, symbol: str):
    """Plot cumulative returns: model strategy vs buy & hold.

    Args:
        bt_df: Backtest DataFrame with BuyHold_Cum and Strategy_Cum columns.
        symbol: Cryptocurrency ticker for the title.
    """
    if bt_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt_df['Date'], bt_df['BuyHold_Cum'], label='Buy & Hold',
            color='#4e79a7', linewidth=2)
    ax.plot(bt_df['Date'], bt_df['Strategy_Cum'], label='Model Strategy',
            color='#e15759', linewidth=2)
    ax.fill_between(bt_df['Date'], bt_df['BuyHold_Cum'],
                    bt_df['Strategy_Cum'], alpha=0.15, color='gray')
    ax.set_title(f'Backtesting — {symbol} (Test Period)', fontsize=14)
    ax.set_ylabel('Cumulative Return (1 = starting capital)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '23_scoring_backtest.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 23_scoring_backtest.png')


# ═══════════════════════════════════════════════════════════════════════════
# 7. PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _make_rl_table(df: pd.DataFrame, title: str = ''):
    """Convert a DataFrame into a ReportLab Table flowable.

    Args:
        df: DataFrame to render.
        title: Optional title for the table.

    Returns:
        List of flowable elements (title paragraph + table).
    """
    styles = getSampleStyleSheet()
    elements = []

    if title:
        elements.append(Paragraph(title, styles['Heading3']))
        elements.append(Spacer(1, 6))

    # Build table data — header + rows
    header = [str(c) for c in [''] + list(df.columns)] if df.index.name or True \
        else [str(c) for c in df.columns]
    data = [header]
    for idx, row in df.iterrows():
        data.append([str(idx)] + [str(v) for v in row.values])

    # Dynamic column widths
    n_cols = len(header)
    col_width = min(1.1 * inch, (7.0 * inch) / n_cols)

    tbl = Table(data, colWidths=[col_width] * n_cols)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTSIZE', (0, 1), (-1, -1), 6.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#ecf0f1'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(tbl)
    return elements


def generate_pdf_report(clf_df, reg_metrics, clust_df, bt_summary,
                        bt_symbols):
    """Assemble the final PDF report with tables and embedded figures.

    Args:
        clf_df: Classification results DataFrame.
        reg_metrics: Regression metrics dict.
        clust_df: Clustering metrics DataFrame.
        bt_summary: List of backtesting summary dicts.
        bt_symbols: List of symbols backtested.
    """
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=20, textColor=colors.HexColor('#2c3e50'),
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', parent=styles['Heading2'],
        fontSize=14, spaceAfter=10, textColor=colors.HexColor('#34495e'),
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=10, spaceAfter=8, leading=14,
    )

    elements = []

    # ── Title page ──────────────────────────────────────────────────────────
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph('🏆 Crypto ML Project — Scoring Report',
                              title_style))
    elements.append(Paragraph('CortisolTeam — Final Model Evaluation',
                              subtitle_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        body_style
    ))
    elements.append(Paragraph(
        'This report consolidates the evaluation of all supervised and '
        'unsupervised models trained on cryptocurrency historical price data. '
        'It includes classification, regression, and clustering metrics, '
        'confusion matrices, ROC curves, and a simple backtesting simulation.',
        body_style
    ))
    elements.append(PageBreak())

    # ── 1. Classification ───────────────────────────────────────────────────
    elements.append(Paragraph('1. Classification Models', subtitle_style))
    elements.append(Paragraph(
        'Three classifiers were trained to predict whether the next-day close '
        'price will increase (1) or decrease (0). Evaluation uses temporal '
        'train/val/test split (80/10/10) with TimeSeriesSplit cross-validation.',
        body_style
    ))

    # Subset columns for readable table
    clf_display = clf_df[[
        'Test_Accuracy', 'Test_Precision', 'Test_Recall',
        'Test_F1', 'Test_ROC_AUC'
    ]].copy()
    clf_display.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    elements.extend(_make_rl_table(clf_display,
                                   'Classification — Test Set Metrics'))
    elements.append(Spacer(1, 12))

    # Figures
    fig_path = FIGURES_DIR / '17_scoring_clf_comparison.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=6 * inch,
                                height=3 * inch))
        elements.append(Spacer(1, 8))

    fig_path = FIGURES_DIR / '18_scoring_confusion_matrices.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=6.5 * inch,
                                height=2.2 * inch))
        elements.append(Spacer(1, 8))

    fig_path = FIGURES_DIR / '19_scoring_roc_curves.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=5 * inch,
                                height=3.5 * inch))

    elements.append(PageBreak())

    # ── 2. Regression ───────────────────────────────────────────────────────
    elements.append(Paragraph('2. Regression Model', subtitle_style))
    elements.append(Paragraph(
        'A RandomForestRegressor was trained to predict the next-day close '
        'price. Metrics include MAE, RMSE, MAPE, and R².',
        body_style
    ))

    reg_display = pd.DataFrame([{
        'MAE': reg_metrics.get('Test_MAE', ''),
        'RMSE': reg_metrics.get('Test_RMSE', ''),
        'MAPE (%)': reg_metrics.get('Test_MAPE', ''),
        'R²': reg_metrics.get('Test_R2', ''),
        'CV R² mean': reg_metrics.get('CV_R2_mean', ''),
    }], index=['RFRegressor'])
    elements.extend(_make_rl_table(reg_display,
                                   'Regression — Test Set Metrics'))
    elements.append(Spacer(1, 12))

    fig_path = FIGURES_DIR / '20_scoring_regression_scatter.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=5 * inch,
                                height=3.5 * inch))
        elements.append(Spacer(1, 8))

    fig_path = FIGURES_DIR / '21_scoring_residuals.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=6 * inch,
                                height=2.5 * inch))

    elements.append(PageBreak())

    # ── 3. Clustering ───────────────────────────────────────────────────────
    elements.append(Paragraph('3. Clustering Models', subtitle_style))
    elements.append(Paragraph(
        'Four clustering configurations were evaluated using Silhouette Score, '
        'Davies-Bouldin Index, and Calinski-Harabasz Index. Features used: '
        'mean_return, volatility, sharpe_ratio, max_drawdown, avg_volume.',
        body_style
    ))

    if not clust_df.empty:
        clust_display = clust_df.set_index('Model')[[
            'N_Clusters', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz'
        ]].copy()
        clust_display.columns = ['K', 'Silhouette', 'Davies-Bouldin',
                                 'Calinski-Harabasz']
        elements.extend(_make_rl_table(clust_display,
                                       'Clustering Quality Metrics'))
        elements.append(Spacer(1, 12))

        fig_path = FIGURES_DIR / '22_scoring_clustering_metrics.png'
        if fig_path.exists():
            elements.append(RLImage(str(fig_path), width=6 * inch,
                                    height=2.5 * inch))

    elements.append(PageBreak())

    # ── 4. Backtesting ──────────────────────────────────────────────────────
    elements.append(Paragraph('4. Backtesting Simulation', subtitle_style))
    elements.append(Paragraph(
        'A simple simulation compares a strategy based on classifier '
        'predictions (buy when prediction=1, sell when prediction=0) against '
        'a passive buy & hold strategy over the test period.',
        body_style
    ))

    if bt_summary:
        bt_display = pd.DataFrame(bt_summary).set_index('Symbol')
        elements.extend(_make_rl_table(bt_display, 'Backtesting Results'))
        elements.append(Spacer(1, 12))

    fig_path = FIGURES_DIR / '23_scoring_backtest.png'
    if fig_path.exists():
        elements.append(RLImage(str(fig_path), width=6 * inch,
                                height=2.5 * inch))

    elements.append(PageBreak())

    # ── 5. Consolidated Summary ─────────────────────────────────────────────
    elements.append(Paragraph('5. Consolidated Model Comparison',
                              subtitle_style))
    elements.append(Paragraph(
        'The table below summarises the key metric for each model type '
        'to enable a quick cross-comparison.',
        body_style
    ))

    summary_rows = []
    for model in clf_df.index:
        summary_rows.append({
            'Model': model, 'Type': 'Classification',
            'Key Metric': f"ROC-AUC = {clf_df.loc[model, 'Test_ROC_AUC']}"
        })
    summary_rows.append({
        'Model': 'RFRegressor', 'Type': 'Regression',
        'Key Metric': f"R² = {reg_metrics.get('Test_R2', 'N/A')}"
    })
    if not clust_df.empty:
        for _, row in clust_df.iterrows():
            summary_rows.append({
                'Model': row['Model'], 'Type': 'Clustering',
                'Key Metric': f"Silhouette = {row.get('Silhouette', 'N/A')}"
            })

    summary_df = pd.DataFrame(summary_rows).set_index('Model')
    elements.extend(_make_rl_table(summary_df, 'All Models at a Glance'))
    elements.append(Spacer(1, 20))

    elements.append(HRFlowable(
        width='100%', thickness=1, color=colors.HexColor('#bdc3c7')
    ))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        'End of report. Generated automatically by src/scoring.py.',
        body_style
    ))

    # Build PDF
    doc.build(elements)
    print(f'\n[pdf] Report saved → {PDF_PATH}')


# ═══════════════════════════════════════════════════════════════════════════
# 8. METRICS SUMMARY CSV
# ═══════════════════════════════════════════════════════════════════════════

def save_metrics_csv(clf_df, reg_metrics, clust_df, bt_summary):
    """Save a consolidated CSV with all metrics for every model.

    Args:
        clf_df: Classification metrics DataFrame.
        reg_metrics: Regression metrics dict.
        clust_df: Clustering metrics DataFrame.
        bt_summary: List of backtesting summary dicts.
    """
    rows = []

    # Classification rows
    for model in clf_df.index:
        row = clf_df.loc[model].to_dict()
        row['Model'] = model
        rows.append(row)

    # Regression row
    rows.append(reg_metrics)

    # Clustering rows
    if not clust_df.empty:
        for _, r in clust_df.iterrows():
            rows.append(r.to_dict())

    # Backtesting rows
    for bt in bt_summary:
        bt_row = bt.copy()
        bt_row['Type'] = 'Backtesting'
        bt_row['Model'] = f"Backtest_{bt['Symbol']}"
        rows.append(bt_row)

    out = pd.DataFrame(rows)

    # Reorder columns: Model and Type first
    priority = ['Model', 'Type']
    other_cols = [c for c in out.columns if c not in priority]
    out = out[priority + other_cols]

    out.to_csv(CSV_PATH, index=False)
    print(f'[csv] Metrics saved → {CSV_PATH}')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Execute the complete scoring pipeline."""
    print('=' * 60)
    print('[scoring] 🏆 Starting scoring pipeline ...')
    print('=' * 60)

    # ── Load data ───────────────────────────────────────────────────────────
    print('\n[1/7] Loading features ...')
    df = load_features(DATA_PATH)
    print(f'  Shape: {df.shape}  |  Coins: {df["Symbol"].nunique()}')
    print(f'  Target balance: {df[TARGET_CLF].value_counts().to_dict()}')

    # ── Temporal split ──────────────────────────────────────────────────────
    print('\n[2/7] Temporal split 80/10/10 ...')
    train, val, test = split_temporal(df)

    # ── Classification ──────────────────────────────────────────────────────
    print('\n[3/7] Training & evaluating classifiers ...')
    clf_df, best_clf, best_clf_name = train_classifiers(train, val, test)
    print('\n  Classification metrics (test set):')
    print(clf_df.to_string())

    # Save best classifier
    joblib.dump(best_clf, MODELS_DIR / 'best_classifier.pkl')
    print(f'  → Saved models/best_classifier.pkl ({best_clf_name})')

    # ── Regression ──────────────────────────────────────────────────────────
    print('\n[4/7] Training & evaluating regressor ...')
    reg_metrics, reg_pipe = train_regressor(train, val, test)
    print('\n  Regression metrics:')
    for k, v in reg_metrics.items():
        if k not in ('Model', 'Type'):
            print(f'    {k:<15}: {v}')

    joblib.dump(reg_pipe, MODELS_DIR / 'best_regressor.pkl')
    print('  → Saved models/best_regressor.pkl')

    # ── Clustering ──────────────────────────────────────────────────────────
    print('\n[5/7] Evaluating clustering algorithms ...')
    clust_df = evaluate_clustering()
    if not clust_df.empty:
        print('\n  Clustering metrics:')
        print(clust_df.to_string(index=False))

    # ── Backtesting ─────────────────────────────────────────────────────────
    print('\n[6/7] Backtesting simulation ...')
    bt_symbols = ['BTC', 'ETH']
    bt_summaries = []
    all_fitted_clf = build_classifiers()
    # Re-fit all for confusion matrices
    X_tr = train[FEATURE_COLS].values
    y_tr = train[TARGET_CLF].values
    for name, pipe in all_fitted_clf.items():
        pipe.fit(X_tr, y_tr)

    for sym in bt_symbols:
        bt_coin, bt_sum = backtest(test, best_clf, symbol=sym)
        if bt_sum:
            bt_summaries.append(bt_sum)
            print(f'\n  {sym} backtest:')
            for k, v in bt_sum.items():
                print(f'    {k:<25}: {v}')

    # ── Figures ─────────────────────────────────────────────────────────────
    print('\n[7/7] Generating figures ...')
    X_te = test[FEATURE_COLS].values
    y_te_clf = test[TARGET_CLF].values
    y_te_reg = test[TARGET_REG].values

    plot_classification_comparison(clf_df)
    plot_confusion_matrices(all_fitted_clf, X_te, y_te_clf)
    plot_roc_curves(all_fitted_clf, X_te, y_te_clf)
    plot_regression_scatter(reg_pipe, X_te, y_te_reg)
    plot_regression_residuals(reg_pipe, X_te, y_te_reg)
    plot_clustering_metrics(clust_df)

    # Backtest plot for first symbol
    if bt_symbols:
        bt_coin, _ = backtest(test, best_clf, symbol=bt_symbols[0])
        plot_backtest(bt_coin, bt_symbols[0])

    # ── Save CSV ────────────────────────────────────────────────────────────
    print('\n[scoring] Saving metrics CSV ...')
    save_metrics_csv(clf_df, reg_metrics, clust_df, bt_summaries)

    # ── Generate PDF ────────────────────────────────────────────────────────
    print('\n[scoring] Generating PDF report ...')
    generate_pdf_report(clf_df, reg_metrics, clust_df, bt_summaries,
                        bt_symbols)

    print('\n' + '=' * 60)
    print('[done] 🏆 Scoring pipeline completed successfully!')
    print(f'  → {CSV_PATH}')
    print(f'  → {PDF_PATH}')
    print('=' * 60)


if __name__ == '__main__':
    main()

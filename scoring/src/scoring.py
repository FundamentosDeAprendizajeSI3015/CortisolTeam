"""Scoring pipeline.

Evaluates classification, regression, and clustering models, generates
figures, consolidates metrics, runs a simple backtest, and exports a PDF.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH_LOCAL = SCORE_ROOT / "data" / "processed" / "features.parquet"
DATA_PATH_REPO = REPO_ROOT / "data" / "processed" / "features.parquet"
DATA_PATH = DATA_PATH_LOCAL if DATA_PATH_LOCAL.exists() else DATA_PATH_REPO

CLUST_FEAT_LOCAL = SCORE_ROOT / "data" / "features_clustering.csv"
CLUST_FEAT_REPO = REPO_ROOT / "data" / "features_clustering.csv"
CLUST_FEAT = CLUST_FEAT_LOCAL if CLUST_FEAT_LOCAL.exists() else CLUST_FEAT_REPO

MODELS_DIR = SCORE_ROOT / "models"
REPORTS_DIR = SCORE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDF_PATH = REPORTS_DIR / "model_report.pdf"
CSV_PATH = REPORTS_DIR / "metrics_summary.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURE_COLS = [
    "sma_7", "sma_14", "sma_30",
    "ema_14", "rsi_14",
    "bb_high", "bb_low", "bb_width",
    "macd", "macd_signal",
    "Return",
]
LAG_COLS_SOURCE = ["Return", "rsi_14", "macd"]
LAG_PERIODS = [1, 3, 7]

TARGET_CLF = "target"
TARGET_REG = "Return"
RETURN_MIN = -1.0
RETURN_MAX = 5.0
RANDOM_STATE = 42

FIG_CLF_COMP = FIGURES_DIR / "17_scoring_clf_comparison.png"
FIG_CONFUSION = FIGURES_DIR / "18_scoring_confusion_matrices.png"
FIG_ROC = FIGURES_DIR / "19_scoring_roc_curves.png"
FIG_REG_SCATTER = FIGURES_DIR / "20_scoring_regression_scatter.png"
FIG_RESIDUALS = FIGURES_DIR / "21_scoring_residuals.png"
FIG_CLUSTER = FIGURES_DIR / "22_scoring_clustering_metrics.png"
FIG_BACKTEST = FIGURES_DIR / "23_scoring_backtest.png"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LAG_COLS_SOURCE:
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = df.groupby("Symbol")[col].shift(lag)
    return df


def build_feature_cols(df: pd.DataFrame) -> list:
    lag_cols = [
        f"{col}_lag{lag}"
        for col in LAG_COLS_SOURCE
        for lag in LAG_PERIODS
        if f"{col}_lag{lag}" in df.columns
    ]
    return BASE_FEATURE_COLS + lag_cols


def load_features(path: Path) -> tuple:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run supervised/feature_engineering.py first."
        )

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    required = set(BASE_FEATURE_COLS) | {TARGET_CLF, TARGET_REG, "Date", "Symbol", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in features.parquet: {missing}")

    df = add_lag_features(df)
    feature_cols = build_feature_cols(df)

    n_before = len(df)
    df = df.dropna(subset=feature_cols + [TARGET_CLF, TARGET_REG])
    print(f"  Rows dropped due to NaN: {n_before - len(df)}")

    n_before = len(df)
    df = df[df[TARGET_REG].between(RETURN_MIN, RETURN_MAX)].copy()
    print(f"  Rows dropped due to Return outliers: {n_before - len(df)}")

    return df, feature_cols


def split_temporal(df: pd.DataFrame, val_size: float = 0.10, test_size: float = 0.10):
    dates = df["Date"].sort_values().unique()
    n = len(dates)
    cut_val = dates[int(n * (1 - val_size - test_size))]
    cut_test = dates[int(n * (1 - test_size))]

    train = df[df["Date"] < cut_val].copy()
    val = df[(df["Date"] >= cut_val) & (df["Date"] < cut_test)].copy()
    test = df[df["Date"] >= cut_test].copy()

    print(
        f"  Train: {len(train):>6} rows  ({train['Date'].min().date()} -> {train['Date'].max().date()})"
    )
    print(
        f"  Val  : {len(val):>6} rows  ({val['Date'].min().date()} -> {val['Date'].max().date()})"
    )
    print(
        f"  Test : {len(test):>6} rows  ({test['Date'].min().date()} -> {test['Date'].max().date()})"
    )
    return train, val, test


def build_classifiers() -> dict:
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
            )),
        ]),
        "RandomForestClassifier": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            )),
        ]),
    }

    if HAS_XGB:
        models["XGBClassifier"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, eval_metric="logloss",
                verbosity=0,
            )),
        ])

    return models


def build_regressors() -> dict:
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge()),
        ]),
        "RandomForestRegressor": Pipeline([
            ("reg", RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }

    if HAS_XGB:
        models["XGBRegressor"] = Pipeline([
            ("reg", XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0,
            )),
        ])

    return models


def train_classifiers(train, val, test, feature_cols):
    X_tr, y_tr = train[feature_cols].values, train[TARGET_CLF].values
    X_va, y_va = val[feature_cols].values, val[TARGET_CLF].values
    X_te, y_te = test[feature_cols].values, test[TARGET_CLF].values

    tscv = TimeSeriesSplit(n_splits=5)
    models = build_classifiers()
    results = []
    fitted = {}

    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=tscv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe

        y_va_pred = pipe.predict(X_va)
        y_te_pred = pipe.predict(X_te)

        try:
            y_va_proba = pipe.predict_proba(X_va)[:, 1]
            y_te_proba = pipe.predict_proba(X_te)[:, 1]
            auc_va = roc_auc_score(y_va, y_va_proba)
            auc_te = roc_auc_score(y_te, y_te_proba)
        except Exception:
            auc_va = float("nan")
            auc_te = float("nan")

        results.append({
            "Model": name,
            "Type": "Classification",
            "CV_AUC_mean": round(float(cv_scores.mean()), 4),
            "CV_AUC_std": round(float(cv_scores.std()), 4),
            "Val_Accuracy": round(float(accuracy_score(y_va, y_va_pred)), 4),
            "Val_Precision": round(float(precision_score(y_va, y_va_pred, zero_division=0)), 4),
            "Val_Recall": round(float(recall_score(y_va, y_va_pred, zero_division=0)), 4),
            "Val_F1": round(float(f1_score(y_va, y_va_pred, zero_division=0)), 4),
            "Val_ROC_AUC": round(float(auc_va), 4),
            "Test_Accuracy": round(float(accuracy_score(y_te, y_te_pred)), 4),
            "Test_Precision": round(float(precision_score(y_te, y_te_pred, zero_division=0)), 4),
            "Test_Recall": round(float(recall_score(y_te, y_te_pred, zero_division=0)), 4),
            "Test_F1": round(float(f1_score(y_te, y_te_pred, zero_division=0)), 4),
            "Test_ROC_AUC": round(float(auc_te), 4),
        })

    results_df = pd.DataFrame(results).set_index("Model")
    best_name = results_df["Val_ROC_AUC"].fillna(-np.inf).idxmax()
    best_pipe = fitted[best_name]

    return results_df, best_pipe, best_name, fitted


def train_regressors(train, val, test, feature_cols):
    feature_cols_reg = [
        c for c in feature_cols
        if c != TARGET_REG and not c.startswith("Return_lag")
    ]

    X_tr, y_tr = train[feature_cols_reg].values, train[TARGET_REG].values
    X_va, y_va = val[feature_cols_reg].values, val[TARGET_REG].values
    X_te, y_te = test[feature_cols_reg].values, test[TARGET_REG].values

    tscv = TimeSeriesSplit(n_splits=5)
    models = build_regressors()
    results = []
    fitted = {}

    def mape(y_true, y_pred):
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=tscv, scoring="r2", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe

        y_va_pred = pipe.predict(X_va)
        y_te_pred = pipe.predict(X_te)

        results.append({
            "Model": name,
            "Type": "Regression",
            "CV_R2_mean": round(float(cv_scores.mean()), 4),
            "CV_R2_std": round(float(cv_scores.std()), 4),
            "Val_MAE": round(float(mean_absolute_error(y_va, y_va_pred)), 6),
            "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_va, y_va_pred))), 6),
            "Val_MAPE": round(mape(y_va, y_va_pred), 4),
            "Val_R2": round(float(r2_score(y_va, y_va_pred)), 4),
            "Test_MAE": round(float(mean_absolute_error(y_te, y_te_pred)), 6),
            "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_te, y_te_pred))), 6),
            "Test_MAPE": round(mape(y_te, y_te_pred), 4),
            "Test_R2": round(float(r2_score(y_te, y_te_pred)), 4),
        })

    results_df = pd.DataFrame(results).set_index("Model")
    best_name = results_df["Val_R2"].idxmax()
    best_pipe = fitted[best_name]

    return results_df, best_pipe, best_name, fitted, feature_cols_reg


def evaluate_clustering():
    if not CLUST_FEAT.exists():
        print(f"  [warn] {CLUST_FEAT} not found. Skipping clustering eval.")
        return pd.DataFrame()

    features = pd.read_csv(CLUST_FEAT, index_col="Symbol")
    X = features.values

    algorithms = {
        "KMeans_K2": KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10),
        "KMeans_K4": KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10),
        "Agglomerative_K2": AgglomerativeClustering(n_clusters=2),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=2),
    }

    rows = []
    for name, algo in algorithms.items():
        labels = algo.fit_predict(X)
        n_clusters = len(set(labels) - {-1})

        if n_clusters < 2:
            rows.append({
                "Model": name,
                "Type": "Clustering",
                "N_Clusters": n_clusters,
                "Silhouette": float("nan"),
                "Davies_Bouldin": float("nan"),
                "Calinski_Harabasz": float("nan"),
            })
            continue

        mask = labels != -1
        X_eval = X[mask] if name == "DBSCAN" else X
        labels_eval = labels[mask] if name == "DBSCAN" else labels

        rows.append({
            "Model": name,
            "Type": "Clustering",
            "N_Clusters": n_clusters,
            "Silhouette": round(float(silhouette_score(X_eval, labels_eval)), 4),
            "Davies_Bouldin": round(float(davies_bouldin_score(X_eval, labels_eval)), 4),
            "Calinski_Harabasz": round(float(calinski_harabasz_score(X_eval, labels_eval)), 4),
        })

    return pd.DataFrame(rows).set_index("Model")


def backtest(test: pd.DataFrame, classifier_pipe, feature_cols: list, symbol: str = "BTC"):
    coin = test[test["Symbol"] == symbol].copy().sort_values("Date")
    if len(coin) < 2:
        print(f"  [warn] Not enough data for {symbol} in test set.")
        return pd.DataFrame(), {}

    X_coin = coin[feature_cols].values
    preds = classifier_pipe.predict(X_coin)

    coin = coin.reset_index(drop=True)
    coin["Prediction"] = preds
    coin["Daily_Return"] = coin["Close"].pct_change()
    coin["Strategy_Return"] = coin["Daily_Return"] * coin["Prediction"]
    coin["BuyHold_Cum"] = (1 + coin["Daily_Return"]).cumprod()
    coin["Strategy_Cum"] = (1 + coin["Strategy_Return"]).cumprod()
    coin["BuyHold_Cum"] = coin["BuyHold_Cum"].fillna(1.0)
    coin["Strategy_Cum"] = coin["Strategy_Cum"].fillna(1.0)

    final_bh = coin["BuyHold_Cum"].iloc[-1]
    final_st = coin["Strategy_Cum"].iloc[-1]
    total_days = len(coin)
    days_in = int(coin["Prediction"].sum())

    summary = {
        "Symbol": symbol,
        "Test_Days": total_days,
        "Days_In_Market": days_in,
        "Days_Out_Market": total_days - days_in,
        "BuyHold_Total_Return": round(float((final_bh - 1) * 100), 2),
        "Strategy_Total_Return": round(float((final_st - 1) * 100), 2),
        "Strategy_vs_BuyHold_pct": round(float((final_st / final_bh - 1) * 100), 2),
    }
    return coin, summary


def plot_clf_comparison(results_df: pd.DataFrame):
    metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]
    data = results_df[metrics]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(data.index))
    width = 0.15

    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, data[metric], width, label=metric.replace("Test_", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classification Models - Test Set Metrics Comparison")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIG_CLF_COMP, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(pipes: dict, X_test, y_test):
    names = list(pipes.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        y_pred = pipes[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Down (0)", "Up (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    fig.suptitle("Confusion Matrices - Test Set")
    plt.tight_layout()
    fig.savefig(FIG_CONFUSION, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(pipes: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, pipe in pipes.items():
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate (Positive label: 1)")
    ax.set_ylabel("True Positive Rate (Positive label: 1)")
    ax.set_title("ROC Curves - All Classifiers (Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG_ROC, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_scatter(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15)
    lim = max(abs(y_test).max(), abs(y_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title("Regression - Predicted vs Actual (Test Set)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_REG_SCATTER, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(residuals, bins=60, color="teal", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")

    axes[1].scatter(y_pred, residuals, alpha=0.4, s=12, color="tomato")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Residual")

    plt.tight_layout()
    fig.savefig(FIG_RESIDUALS, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_metrics(clust_df: pd.DataFrame):
    if clust_df.empty:
        return

    df = clust_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(df.index, df["Silhouette"], color="green")
    axes[0].set_title("Silhouette Score (higher is better)")
    axes[0].set_ylim(0, max(0.1, np.nanmax(df["Silhouette"]) + 0.1))

    axes[1].bar(df.index, df["Davies_Bouldin"], color="red")
    axes[1].set_title("Davies-Bouldin Index (lower is better)")

    axes[2].bar(df.index, df["Calinski_Harabasz"], color="steelblue")
    axes[2].set_title("Calinski-Harabasz Index (higher is better)")

    for ax in axes:
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig.savefig(FIG_CLUSTER, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(bt_df: pd.DataFrame, symbol: str):
    if bt_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(bt_df["Date"], bt_df["BuyHold_Cum"], label="Buy & Hold", color="steelblue")
    ax.plot(bt_df["Date"], bt_df["Strategy_Cum"], label="Model Strategy", color="tomato")
    ax.fill_between(bt_df["Date"], bt_df["BuyHold_Cum"], color="gray", alpha=0.08)
    ax.set_title(f"Backtesting - {symbol} (Test Period)")
    ax.set_ylabel("Cumulative Return (1 = starting capital)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_BACKTEST, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(clf_df, reg_df, clust_df, bt_summary):
    rows = []

    if isinstance(clf_df, pd.DataFrame) and not clf_df.empty:
        for model in clf_df.index:
            row = clf_df.loc[model].to_dict()
            row["Model"] = model
            row["Type"] = "Classification"
            rows.append(row)

    if isinstance(reg_df, pd.DataFrame) and not reg_df.empty:
        for model in reg_df.index:
            row = reg_df.loc[model].to_dict()
            row["Model"] = model
            row["Type"] = "Regression"
            rows.append(row)

    if isinstance(clust_df, pd.DataFrame) and not clust_df.empty:
        for model in clust_df.index:
            row = clust_df.loc[model].to_dict()
            row["Model"] = model
            row["Type"] = "Clustering"
            rows.append(row)

    for bt in bt_summary:
        row = bt.copy()
        row["Model"] = f"Backtest_{bt['Symbol']}"
        row["Type"] = "Backtesting"
        rows.append(row)

    out = pd.DataFrame(rows)
    priority = ["Model", "Type"]
    other_cols = [c for c in out.columns if c not in priority]
    out = out[priority + other_cols]
    out.to_csv(CSV_PATH, index=False)
    print(f"[csv] Metrics saved -> {CSV_PATH}")
    return out


def _format_table(df: pd.DataFrame, float_cols: list, int_cols: list | None = None):
    table_df = df.copy()
    int_cols = int_cols or []

    for col in int_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(
                lambda x: "" if pd.isna(x) else f"{int(x)}"
            )

    for col in float_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.4f}"
            )

    return table_df


def _df_to_reportlab_table(df: pd.DataFrame, col_widths=None):
    """Convierte un DataFrame a un formato de tabla para reportlab."""
    data = [list(df.columns)] + df.values.tolist()
    
    if col_widths is None:
        # Calcular ancho automático basado en número de columnas
        num_cols = len(df.columns)
        if num_cols <= 3:
            col_widths = [2.2 * inch for _ in range(num_cols)]
        elif num_cols <= 5:
            col_widths = [1.4 * inch for _ in range(num_cols)]
        else:
            col_widths = [1.0 * inch for _ in range(num_cols)]
    
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    return table


def generate_pdf_report(res_clf, res_reg, clust_df, bt_summary, best_clf, best_reg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Preparar estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    footer_style = ParagraphStyle(
        'CustomFooter',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_LEFT,
        textColor=colors.grey,
        spaceAfter=4,
    )
    
    # Crear lista de elementos para el PDF
    elements = []
    
    # Portada
    elements.append(Paragraph("■ Proyecto ML Cripto — Reporte de Puntuación", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("CortisolTeam — Evaluación Final de Modelos", subtitle_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Generado: {now}", normal_style))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "Este reporte consolida la evaluación de todos los modelos supervisados y no supervisados entrenados en "
        "datos históricos de precios de criptomonedas. Incluye métricas de clasificación, regresión y agrupamiento, "
        "matrices de confusión, curvas ROC y una simulación simple de backtesting.",
        normal_style
    ))
    elements.append(PageBreak())
    
    # Sección 1: Classification
    clf_table = res_clf[[
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC",
    ]].reset_index().rename(columns={
        "Test_Accuracy": "Exactitud",
        "Test_Precision": "Precisión",
        "Test_Recall": "Recall",
        "Test_F1": "F1",
        "Test_ROC_AUC": "ROC-AUC",
        "Model": "Modelo",
    })
    clf_table = _format_table(
        clf_table,
        float_cols=["Exactitud", "Precisión", "Recall", "F1", "ROC-AUC"],
    )
    clf_rl_table = _df_to_reportlab_table(clf_table, col_widths=[1.4*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    
    elements.append(Paragraph("1. Modelos de Clasificación", section_style))
    elements.append(Paragraph(
        "Tres clasificadores fueron entrenados para predecir si el precio de cierre del día siguiente aumentará (1) o disminuirá (0). "
        "La evaluación utiliza un split temporal train/val/test (80/10/10) con validación cruzada TimeSeriesSplit.",
        normal_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Clasificación — Métricas del Conjunto de Prueba", ParagraphStyle(
        'TableTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=6
    )))
    elements.append(clf_rl_table)
    elements.append(Spacer(1, 12))
    
    if FIG_CLF_COMP.exists():
        img = Image(str(FIG_CLF_COMP), width=6*inch, height=3*inch)
        if img.filename:
            elements.append(img)
            elements.append(Spacer(1, 6))
    
    if FIG_CONFUSION.exists():
        img = Image(str(FIG_CONFUSION), width=6.5*inch, height=3*inch)
        if img.filename:
            elements.append(img)
            elements.append(Spacer(1, 6))
    
    if FIG_ROC.exists():
        img = Image(str(FIG_ROC), width=5.5*inch, height=3*inch)
        if img.filename:
            elements.append(img)
    
    elements.append(PageBreak())
    
    # Sección 2: Regression
    reg_table = res_reg[[
        "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_R2", "CV_R2_mean",
    ]].reset_index().rename(columns={
        "Test_MAE": "MAE",
        "Test_RMSE": "RMSE",
        "Test_MAPE": "MAPE (%)",
        "Test_R2": "R²",
        "CV_R2_mean": "Media CV R²",
        "Model": "Modelo",
    })
    reg_table = _format_table(
        reg_table,
        float_cols=["MAE", "RMSE", "MAPE (%)", "R²", "Media CV R²"],
    )
    reg_rl_table = _df_to_reportlab_table(reg_table, col_widths=[1.3*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.3*inch])
    
    elements.append(Paragraph("2. Modelo de Regresión", section_style))
    elements.append(Paragraph(
        "Un RandomForestRegressor fue entrenado para predecir el precio de cierre del día siguiente. Las métricas incluyen MAE, "
        "RMSE, MAPE y R².",
        normal_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Regresión — Métricas del Conjunto de Prueba", ParagraphStyle(
        'TableTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=6
    )))
    elements.append(reg_rl_table)
    elements.append(Spacer(1, 12))
    
    if FIG_REG_SCATTER.exists():
        img = Image(str(FIG_REG_SCATTER), width=5.5*inch, height=3.5*inch)
        if img.filename:
            elements.append(img)
            elements.append(Spacer(1, 6))
    
    if FIG_RESIDUALS.exists():
        img = Image(str(FIG_RESIDUALS), width=6.5*inch, height=2.5*inch)
        if img.filename:
            elements.append(img)
    
    elements.append(PageBreak())
    
    # Sección 3: Clustering
    clust_table = pd.DataFrame()
    clust_rl_table = None
    if not clust_df.empty:
        clust_table = clust_df[[
            "N_Clusters", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz",
        ]].reset_index().rename(columns={
            "N_Clusters": "K",
            "Silhouette": "Silhouette",
            "Davies_Bouldin": "Davies-Bouldin",
            "Calinski_Harabasz": "Calinski-Harabasz",
            "Model": "Modelo",
        })
        clust_table = _format_table(
            clust_table,
            float_cols=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
            int_cols=["K"],
        )
        clust_rl_table = _df_to_reportlab_table(clust_table, col_widths=[1.4*inch, 1.2*inch, 1.4*inch, 1.6*inch, 1.6*inch])
    
    elements.append(Paragraph("3. Modelos de Agrupamiento", section_style))
    elements.append(Paragraph(
        "Se evaluaron cuatro configuraciones de agrupamiento utilizando Puntuación Silhouette, Índice Davies-Bouldin e "
        "Índice Calinski-Harabasz. Características utilizadas: media_retorno, volatilidad, índice_sharpe, máximo_drawdown, volumen_promedio.",
        normal_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Métricas de Calidad del Agrupamiento", ParagraphStyle(
        'TableTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=6
    )))
    if clust_rl_table is not None:
        elements.append(clust_rl_table)
    elements.append(Spacer(1, 12))
    
    if FIG_CLUSTER.exists():
        img = Image(str(FIG_CLUSTER), width=6.5*inch, height=2.8*inch)
        if img.filename:
            elements.append(img)
    
    elements.append(PageBreak())
    
    # Sección 4: Backtesting
    bt_table = pd.DataFrame(bt_summary) if bt_summary else pd.DataFrame()
    bt_rl_table = None
    if not bt_table.empty:
        bt_cols_order = [
            "Symbol", "Test_Days", "Days_In_Market", "Days_Out_Market",
            "BuyHold_Total_Return", "Strategy_Total_Return", "Strategy_vs_BuyHold_pct",
        ]
        available_cols = [c for c in bt_cols_order if c in bt_table.columns]
        bt_table = bt_table[available_cols]
        bt_table.columns = ["Símbolo", "Días Prueba", "Días Mercado", "Días Fuera", "Retorno B&H (%)", "Retorno Estrategia (%)", "Diferencia (%)"]
        bt_table = _format_table(
            bt_table,
            float_cols=[
                "Retorno B&H (%)", "Retorno Estrategia (%)", "Diferencia (%)",
            ],
            int_cols=["Días Prueba", "Días Mercado", "Días Fuera"],
        )
        bt_rl_table = _df_to_reportlab_table(bt_table, col_widths=[0.9*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.2*inch, 1.4*inch, 1.2*inch])
    
    elements.append(Paragraph("4. Simulación de Backtesting", section_style))
    elements.append(Paragraph(
        "Una simulación simple compara una estrategia basada en predicciones del clasificador (comprar cuando predicción=1, "
        "vender cuando predicción=0) contra una estrategia pasiva de comprar y mantener durante el período de prueba.",
        normal_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Resultados de Backtesting", ParagraphStyle(
        'TableTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=6
    )))
    if bt_rl_table is not None:
        elements.append(bt_rl_table)
    elements.append(Spacer(1, 12))
    
    if FIG_BACKTEST.exists():
        img = Image(str(FIG_BACKTEST), width=6.5*inch, height=3*inch)
        if img.filename:
            elements.append(img)
    
    elements.append(PageBreak())
    
    # Sección 5: Consolidated
    consolidated_rows = []
    for model in res_clf.index:
        metric = res_clf.loc[model, "Test_ROC_AUC"]
        metric_str = "nan" if pd.isna(metric) else f"{float(metric):.4f}"
        consolidated_rows.append({
            "Modelo": model,
            "Tipo": "Clasificación",
            "Métrica Clave": f"ROC-AUC = {metric_str}",
        })
    
    for model in res_reg.index:
        metric = res_reg.loc[model, "Test_R2"]
        metric_str = "nan" if pd.isna(metric) else f"{float(metric):.4f}"
        consolidated_rows.append({
            "Modelo": model,
            "Tipo": "Regresión",
            "Métrica Clave": f"R² = {metric_str}",
        })
    
    if not clust_df.empty:
        for model in clust_df.index:
            metric = clust_df.loc[model, "Silhouette"]
            metric_str = "nan" if pd.isna(metric) else f"{float(metric):.4f}"
            consolidated_rows.append({
                "Modelo": model,
                "Tipo": "Agrupamiento",
                "Métrica Clave": f"Silhouette = {metric_str}",
            })
    
    consolidated_table = pd.DataFrame(consolidated_rows)
    
    elements.append(Paragraph("5. Comparación Consolidada de Modelos", section_style))
    elements.append(Paragraph(
        "La tabla siguiente resume la métrica clave para cada tipo de modelo para permitir una rápida comparación cruzada.",
        normal_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Todos los Modelos de un Vistazo", ParagraphStyle(
        'TableTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=6
    )))
    cons_rl_table = _df_to_reportlab_table(consolidated_table, col_widths=[1.8*inch, 1.8*inch, 2.2*inch])
    elements.append(cons_rl_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Fin del reporte. Generado automáticamente por src/scoring.py.", footer_style))
    
    # Generar PDF
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    
    doc.build(elements)
    print(f"[pdf] Report saved -> {PDF_PATH}")


class Image(RLImage):
    """Wrapper para manejar imágenes que podrían no existir."""
    def __init__(self, filename, width=None, height=None, **kwargs):
        filename = str(filename)
        if not Path(filename).exists():
            # Retorna None si la imagen no existe
            self.filename = None
            return
        try:
            super().__init__(filename, width=width, height=height, **kwargs)
        except Exception as e:
            print(f"[warn] Could not load image {filename}: {e}")
            self.filename = None



def main():
    print("=" * 60)
    print("[scoring] Starting scoring pipeline")
    print("=" * 60)

    print("\n[1/7] Loading features ...")
    df, feature_cols = load_features(DATA_PATH)
    print(f"  Shape: {df.shape} | Coins: {df['Symbol'].nunique()}")
    print(f"  Target balance: {df[TARGET_CLF].value_counts().to_dict()}")

    print("\n[2/7] Temporal split 80/10/10 ...")
    train, val, test = split_temporal(df)

    print("\n[3/7] Training and evaluating classifiers ...")
    res_clf, best_clf, best_clf_name, clf_pipes = train_classifiers(
        train, val, test, feature_cols
    )

    print("\n[4/7] Training and evaluating regressors ...")
    res_reg, best_reg, best_reg_name, reg_pipes, feature_cols_reg = train_regressors(
        train, val, test, feature_cols
    )

    joblib.dump(best_clf, MODELS_DIR / "best_classifier.pkl")
    joblib.dump(best_reg, MODELS_DIR / "best_regressor.pkl")

    print("\n[5/7] Evaluating clustering algorithms ...")
    clust_df = evaluate_clustering()

    print("\n[6/7] Backtesting simulation ...")
    bt_symbols = ["BTC", "ETH"]
    bt_summaries = []
    bt_plot_df = pd.DataFrame()
    bt_plot_symbol = "BTC"
    for sym in bt_symbols:
        bt_df, bt_summary = backtest(test, best_clf, feature_cols, symbol=sym)
        if bt_summary:
            bt_summaries.append(bt_summary)
        if sym == "BTC" and not bt_df.empty:
            bt_plot_df = bt_df
            bt_plot_symbol = sym

    print("\n[7/7] Generating figures and reports ...")
    X_test_clf = test[feature_cols].values
    y_test_clf = test[TARGET_CLF].values

    X_test_reg = test[feature_cols_reg].values
    y_test_reg = test[TARGET_REG].values

    plot_clf_comparison(res_clf)
    plot_confusion_matrices(clf_pipes, X_test_clf, y_test_clf)
    plot_roc_curves(clf_pipes, X_test_clf, y_test_clf)
    plot_regression_scatter(best_reg, X_test_reg, y_test_reg)
    plot_residuals(best_reg, X_test_reg, y_test_reg)
    plot_clustering_metrics(clust_df)
    plot_backtest(bt_plot_df, bt_plot_symbol)

    metrics_df = save_metrics_csv(res_clf, res_reg, clust_df, bt_summaries)
    generate_pdf_report(res_clf, res_reg, clust_df, bt_summaries, best_clf_name, best_reg_name)

    print("\n[done] Scoring pipeline completed")


if __name__ == "__main__":
    main()

"""
Dash Interactive Dashboard - Crypto ML Project

Run:
    python visualization/app/dash_app.py

Notes:
- Uses Plotly + Dash for interactive charts and filters.
- Reads metrics_summary.csv and clustering features.
- If crypto_raw.csv is missing, EDA charts show placeholders.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Paths and data loading
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

CRYPTO_RAW_ENV = os.getenv("CRYPTO_RAW_PATH")
CRYPTO_RAW = Path(CRYPTO_RAW_ENV) if CRYPTO_RAW_ENV else DATA_DIR / "crypto_raw.csv"
FEATURES_PATH = DATA_DIR / "features_clustering.csv"
CLUSTERS_PATH = DATA_DIR / "cluster_labels.csv"
METRICS_PATH = ROOT / "scoring" / "reports" / "metrics_summary.csv"


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#334155"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
    )
    return fig


# Load datasets
DF_METRICS = read_csv_safe(METRICS_PATH)
DF_FEATURES = read_csv_safe(FEATURES_PATH)
DF_CLUSTERS = read_csv_safe(CLUSTERS_PATH)
DF_CRYPTO = read_csv_safe(CRYPTO_RAW)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

CRYPTO_AVAILABLE = False
CRYPTO_SYMBOLS: list[str] = []
CRYPTO_DATE_MIN = None
CRYPTO_DATE_MAX = None

if DF_CRYPTO is not None:
    required_cols = {"Symbol", "Date", "Close"}
    if required_cols.issubset(set(DF_CRYPTO.columns)):
        DF_CRYPTO = DF_CRYPTO.copy()
        DF_CRYPTO["Date"] = pd.to_datetime(DF_CRYPTO["Date"], errors="coerce")
        DF_CRYPTO = DF_CRYPTO.dropna(subset=["Date"]).sort_values(["Symbol", "Date"])
        DF_CRYPTO["Return"] = DF_CRYPTO.groupby("Symbol")["Close"].pct_change()
        DF_CRYPTO["RollingVol30"] = (
            DF_CRYPTO.groupby("Symbol")["Return"].rolling(30).std().reset_index(level=0, drop=True)
        )
        DF_CRYPTO["RollingVol90"] = (
            DF_CRYPTO.groupby("Symbol")["Return"].rolling(90).std().reset_index(level=0, drop=True)
        )
        CRYPTO_AVAILABLE = True
        CRYPTO_SYMBOLS = sorted(DF_CRYPTO["Symbol"].dropna().unique().tolist())
        CRYPTO_DATE_MIN = DF_CRYPTO["Date"].min().date()
        CRYPTO_DATE_MAX = DF_CRYPTO["Date"].max().date()


PCA_DATA = None
CLUSTER_COLUMNS: list[str] = []

if DF_FEATURES is not None and DF_CLUSTERS is not None:
    if "Symbol" in DF_FEATURES.columns and "Symbol" in DF_CLUSTERS.columns:
        CLUSTER_COLUMNS = [c for c in DF_CLUSTERS.columns if c != "Symbol"]
        merged = DF_FEATURES.merge(DF_CLUSTERS, on="Symbol", how="left")
        feature_cols = [c for c in DF_FEATURES.columns if c != "Symbol"]
        if feature_cols:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(merged[feature_cols].fillna(0))
            pca = PCA(n_components=2, random_state=42)
            comps = pca.fit_transform(scaled)
            merged["PC1"] = comps[:, 0]
            merged["PC2"] = comps[:, 1]
            PCA_DATA = merged


# Metrics splits
DF_CLASS = None
DF_REG = None
DF_CLUSTER = None

if DF_METRICS is not None and "Type" in DF_METRICS.columns:
    DF_CLASS = DF_METRICS[DF_METRICS["Type"] == "Classification"].copy()
    DF_REG = DF_METRICS[DF_METRICS["Type"] == "Regression"].copy()
    DF_CLUSTER = DF_METRICS[DF_METRICS["Type"] == "Clustering"].copy()


CLASS_METRICS = {
    "Val ROC AUC": "Val_ROC_AUC",
    "Test ROC AUC": "Test_ROC_AUC",
    "Val Accuracy": "Val_Accuracy",
    "Test Accuracy": "Test_Accuracy",
    "Val F1": "Val_F1",
    "Test F1": "Test_F1",
    "CV AUC Mean": "CV_AUC_mean",
}

REG_METRICS = {
    "Val R2": "Val_R2",
    "Test R2": "Test_R2",
    "Val MAE": "Val_MAE",
    "Test MAE": "Test_MAE",
    "Val RMSE": "Val_RMSE",
    "Test RMSE": "Test_RMSE",
    "CV R2 Mean": "CV_R2_mean",
}

CLUSTER_METRICS = {
    "Silhouette": "Silhouette",
    "Davies Bouldin": "Davies_Bouldin",
    "Calinski Harabasz": "Calinski_Harabasz",
}

CLASS_TABLE_COLS = [
    "Model",
    "CV_AUC_mean",
    "Val_Accuracy",
    "Val_F1",
    "Val_ROC_AUC",
    "Test_Accuracy",
    "Test_F1",
    "Test_ROC_AUC",
]

REG_TABLE_COLS = [
    "Model",
    "CV_R2_mean",
    "Val_MAE",
    "Val_RMSE",
    "Val_R2",
    "Test_MAE",
    "Test_RMSE",
    "Test_R2",
]

CLUSTER_TABLE_COLS = [
    "Model",
    "N_Clusters",
    "Silhouette",
    "Davies_Bouldin",
    "Calinski_Harabasz",
]


def best_model(df: pd.DataFrame | None, metric: str, maximize: bool = True) -> tuple[str, float] | None:
    if df is None or df.empty or metric not in df.columns:
        return None
    temp = df[["Model", metric]].dropna()
    if temp.empty:
        return None
    temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return None
    row = temp.loc[temp[metric].idxmax()] if maximize else temp.loc[temp[metric].idxmin()]
    return row["Model"], float(row[metric])


def format_kpi(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
app = Dash(
    __name__,
    title="Crypto ML Dashboard",
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
)

status_cards = html.Div(
    className="status-grid",
    children=[
        html.Div(
            className=f"status-card {'ok' if CRYPTO_AVAILABLE else 'warn'}",
            children=[
                html.Div("Crypto raw", className="status-title"),
                html.Div("Loaded" if CRYPTO_AVAILABLE else "Missing", className="status-value"),
                html.Div("CRYPTO_RAW_PATH" if CRYPTO_RAW_ENV else "data/crypto_raw.csv", className="status-meta"),
            ],
        ),
        html.Div(
            className=f"status-card {'ok' if DF_FEATURES is not None else 'warn'}",
            children=[
                html.Div("Clustering features", className="status-title"),
                html.Div("Loaded" if DF_FEATURES is not None else "Missing", className="status-value"),
                html.Div("data/features_clustering.csv", className="status-meta"),
            ],
        ),
        html.Div(
            className=f"status-card {'ok' if DF_CLUSTERS is not None else 'warn'}",
            children=[
                html.Div("Cluster labels", className="status-title"),
                html.Div("Loaded" if DF_CLUSTERS is not None else "Missing", className="status-value"),
                html.Div("data/cluster_labels.csv", className="status-meta"),
            ],
        ),
        html.Div(
            className=f"status-card {'ok' if DF_METRICS is not None else 'warn'}",
            children=[
                html.Div("Metrics summary", className="status-title"),
                html.Div("Loaded" if DF_METRICS is not None else "Missing", className="status-value"),
                html.Div("scoring/reports/metrics_summary.csv", className="status-meta"),
            ],
        ),
    ],
)

best_clf = best_model(DF_CLASS, "Test_ROC_AUC", maximize=True)
best_reg = best_model(DF_REG, "Test_R2", maximize=True)
best_cluster = best_model(DF_CLUSTER, "Silhouette", maximize=True)

kpi_cards = html.Div(
    className="kpi-grid",
    children=[
        html.Div(
            className="kpi-card",
            children=[
                html.Div("Best classifier", className="kpi-title"),
                html.Div(best_clf[0] if best_clf else "n/a", className="kpi-value"),
                html.Div(f"Test ROC AUC: {format_kpi(best_clf[1] if best_clf else None)}", className="kpi-meta"),
            ],
        ),
        html.Div(
            className="kpi-card",
            children=[
                html.Div("Best regressor", className="kpi-title"),
                html.Div(best_reg[0] if best_reg else "n/a", className="kpi-value"),
                html.Div(f"Test R2: {format_kpi(best_reg[1] if best_reg else None)}", className="kpi-meta"),
            ],
        ),
        html.Div(
            className="kpi-card",
            children=[
                html.Div("Best clustering", className="kpi-title"),
                html.Div(best_cluster[0] if best_cluster else "n/a", className="kpi-value"),
                html.Div(f"Silhouette: {format_kpi(best_cluster[1] if best_cluster else None)}", className="kpi-meta"),
            ],
        ),
    ],
)


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(
                    className="hero-text",
                    children=[
                        html.H1("Crypto ML Dashboard"),
                        html.P("Interactive model comparisons and performance metrics"),
                        html.Div("Plotly + Dash", className="hero-chip"),
                    ],
                ),
                html.Div(className="hero-note", children="EDA, clustering, supervised models, and scoring in one place."),
            ],
        ),
        status_cards,
        kpi_cards,
        dcc.Tabs(
            className="tabs",
            parent_className="tabs-wrapper",
            children=[
                dcc.Tab(
                    label="EDA",
                    className="tab",
                    selected_className="tab tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.H3("Exploratory data analysis"),
                                        html.P("Interactive views of prices, returns, and volatility."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Symbol"),
                                                dcc.Dropdown(
                                                    id="eda-symbol",
                                                    options=[{"label": s, "value": s} for s in CRYPTO_SYMBOLS],
                                                    value=CRYPTO_SYMBOLS[0] if CRYPTO_SYMBOLS else None,
                                                    clearable=False,
                                                    disabled=not CRYPTO_AVAILABLE,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Date range"),
                                                dcc.DatePickerRange(
                                                    id="eda-date-range",
                                                    min_date_allowed=CRYPTO_DATE_MIN,
                                                    max_date_allowed=CRYPTO_DATE_MAX,
                                                    start_date=CRYPTO_DATE_MIN,
                                                    end_date=CRYPTO_DATE_MAX,
                                                    disabled=not CRYPTO_AVAILABLE,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="graph-grid",
                                    children=[
                                        dcc.Graph(id="eda-price"),
                                        dcc.Graph(id="eda-returns"),
                                        dcc.Graph(id="eda-volatility"),
                                        dcc.Graph(id="eda-volume"),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="No supervisado",
                    className="tab",
                    selected_className="tab tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.H3("Resultados no supervisados"),
                                        html.P("PCA y metricas de clustering."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Cluster label"),
                                                dcc.Dropdown(
                                                    id="cluster-algo",
                                                    options=[{"label": c, "value": c} for c in CLUSTER_COLUMNS],
                                                    value=(
                                                        "KMeans_K4"
                                                        if "KMeans_K4" in CLUSTER_COLUMNS
                                                        else (CLUSTER_COLUMNS[0] if CLUSTER_COLUMNS else None)
                                                    ),
                                                    clearable=False,
                                                    disabled=not CLUSTER_COLUMNS,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Metric"),
                                                dcc.Dropdown(
                                                    id="cluster-metric",
                                                    options=[{"label": k, "value": v} for k, v in CLUSTER_METRICS.items()],
                                                    value=list(CLUSTER_METRICS.values())[0],
                                                    clearable=False,
                                                    disabled=DF_CLUSTER is None,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="graph-grid",
                                    children=[
                                        dcc.Graph(id="cluster-pca"),
                                        dcc.Graph(id="cluster-metrics"),
                                    ],
                                ),
                                dash_table.DataTable(
                                    id="cluster-table",
                                    data=(DF_CLUSTER[CLUSTER_TABLE_COLS].to_dict("records") if DF_CLUSTER is not None else []),
                                    columns=[{"name": c, "id": c} for c in (CLUSTER_TABLE_COLS if DF_CLUSTER is not None else [])],
                                    page_size=6,
                                    style_table={"overflowX": "auto"},
                                    style_header={"backgroundColor": "#0f172a", "color": "white"},
                                    style_cell={"padding": "8px", "fontFamily": "Space Grotesk"},
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Supervisado",
                    className="tab",
                    selected_className="tab tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.H3("Clasificacion"),
                                        html.P("Comparacion de metricas entre validacion y test."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Metric"),
                                                dcc.Dropdown(
                                                    id="clf-metric",
                                                    options=[{"label": k, "value": v} for k, v in CLASS_METRICS.items()],
                                                    value=list(CLASS_METRICS.values())[0],
                                                    clearable=False,
                                                    disabled=DF_CLASS is None,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="clf-metric-graph"),
                                dash_table.DataTable(
                                    id="clf-table",
                                    data=(DF_CLASS[CLASS_TABLE_COLS].to_dict("records") if DF_CLASS is not None else []),
                                    columns=[{"name": c, "id": c} for c in (CLASS_TABLE_COLS if DF_CLASS is not None else [])],
                                    page_size=6,
                                    style_table={"overflowX": "auto"},
                                    style_header={"backgroundColor": "#0f172a", "color": "white"},
                                    style_cell={"padding": "8px", "fontFamily": "Space Grotesk"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="panel-header",
                                    children=[
                                        html.H3("Regresion"),
                                        html.P("Comparacion de metricas entre validacion y test."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Metric"),
                                                dcc.Dropdown(
                                                    id="reg-metric",
                                                    options=[{"label": k, "value": v} for k, v in REG_METRICS.items()],
                                                    value=list(REG_METRICS.values())[0],
                                                    clearable=False,
                                                    disabled=DF_REG is None,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="reg-metric-graph"),
                                dash_table.DataTable(
                                    id="reg-table",
                                    data=(DF_REG[REG_TABLE_COLS].to_dict("records") if DF_REG is not None else []),
                                    columns=[{"name": c, "id": c} for c in (REG_TABLE_COLS if DF_REG is not None else [])],
                                    page_size=6,
                                    style_table={"overflowX": "auto"},
                                    style_header={"backgroundColor": "#0f172a", "color": "white"},
                                    style_cell={"padding": "8px", "fontFamily": "Space Grotesk"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@app.callback(
    Output("eda-price", "figure"),
    Output("eda-returns", "figure"),
    Output("eda-volatility", "figure"),
    Output("eda-volume", "figure"),
    Input("eda-symbol", "value"),
    Input("eda-date-range", "start_date"),
    Input("eda-date-range", "end_date"),
)
def update_eda(symbol: str | None, start_date: str | None, end_date: str | None):
    if not CRYPTO_AVAILABLE or symbol is None:
        message = "crypto_raw.csv not available"
        return (
            empty_figure(message),
            empty_figure(message),
            empty_figure(message),
            empty_figure(message),
        )

    df = DF_CRYPTO[DF_CRYPTO["Symbol"] == symbol].copy()
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if df.empty:
        message = "No data for selected range"
        return (
            empty_figure(message),
            empty_figure(message),
            empty_figure(message),
            empty_figure(message),
        )

    fig_price = px.line(df, x="Date", y="Close", title="Close price")
    fig_price.update_layout(margin=dict(l=30, r=20, t=50, b=30))

    df_returns = df.dropna(subset=["Return"]).copy()
    if df_returns.empty:
        fig_returns = empty_figure("Not enough return data")
    else:
        fig_returns = px.histogram(df_returns, x="Return", nbins=60, title="Return distribution")
        fig_returns.update_layout(margin=dict(l=30, r=20, t=50, b=30))

    df_vol = df.dropna(subset=["RollingVol30", "RollingVol90"], how="all")
    if df_vol.empty:
        fig_vol = empty_figure("Not enough volatility data")
    else:
        fig_vol = px.line(
            df_vol,
            x="Date",
            y=["RollingVol30", "RollingVol90"],
            title="Rolling volatility (30d / 90d)",
        )
        fig_vol.update_layout(margin=dict(l=30, r=20, t=50, b=30))

    if "Volume" not in df.columns:
        fig_volume = empty_figure("Volume column not found")
    else:
        fig_volume = px.area(df, x="Date", y="Volume", title="Trading volume")
        fig_volume.update_layout(margin=dict(l=30, r=20, t=50, b=30))

    return fig_price, fig_returns, fig_vol, fig_volume


@app.callback(
    Output("cluster-pca", "figure"),
    Input("cluster-algo", "value"),
)
def update_cluster_pca(cluster_col: str | None):
    if PCA_DATA is None or not cluster_col or cluster_col not in PCA_DATA.columns:
        return empty_figure("Clustering data not available")

    df_plot = PCA_DATA.copy()
    df_plot["Cluster"] = df_plot[cluster_col].astype(str)
    cluster_values = sorted(df_plot["Cluster"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {val: palette[i % len(palette)] for i, val in enumerate(cluster_values)}

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_map=color_map,
        category_orders={"Cluster": cluster_values},
        hover_data=["Symbol"],
        title=f"PCA projection ({cluster_col})",
    )
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
    return fig


@app.callback(
    Output("cluster-metrics", "figure"),
    Input("cluster-metric", "value"),
)
def update_cluster_metrics(metric: str | None):
    if DF_CLUSTER is None or DF_CLUSTER.empty or metric not in DF_CLUSTER.columns:
        return empty_figure("Cluster metrics not available")

    df = DF_CLUSTER[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("Metric has no values")

    fig = px.bar(df, x="Model", y=metric, title="Clustering metric comparison")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    return fig


@app.callback(
    Output("clf-metric-graph", "figure"),
    Input("clf-metric", "value"),
)
def update_clf_metric(metric: str | None):
    if DF_CLASS is None or DF_CLASS.empty or metric not in DF_CLASS.columns:
        return empty_figure("Classification metrics not available")

    df = DF_CLASS[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("Metric has no values")

    fig = px.bar(df, x="Model", y=metric, title="Classifier comparison")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    return fig


@app.callback(
    Output("reg-metric-graph", "figure"),
    Input("reg-metric", "value"),
)
def update_reg_metric(metric: str | None):
    if DF_REG is None or DF_REG.empty or metric not in DF_REG.columns:
        return empty_figure("Regression metrics not available")

    df = DF_REG[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("Metric has no values")

    fig = px.bar(df, x="Model", y=metric, title="Regressor comparison")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    return fig


if __name__ == "__main__":
    app.run(debug=True)

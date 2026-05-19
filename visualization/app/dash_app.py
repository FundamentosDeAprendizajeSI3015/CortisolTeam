"""
Dashboard interactivo en Dash - Crypto ML Project

Ejecucion:
    python visualization/app/dash_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Rutas y carga de datos
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

CRYPTO_RAW_ENV = os.getenv("CRYPTO_RAW_PATH")
CRYPTO_RAW = Path(CRYPTO_RAW_ENV) if CRYPTO_RAW_ENV else DATA_DIR / "crypto_raw.csv"
FEATURES_PATH = DATA_DIR / "features_clustering.csv"
CLUSTERS_PATH = DATA_DIR / "cluster_labels.csv"
SUPERVISED_REPORT = ROOT / "supervised" / "reports" / "supervised_report.md"


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_lines_safe(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None


def parse_md_row(line: str) -> list[str]:
    return [col.strip() for col in line.strip().strip("|").split("|")]


def parse_markdown_table(lines: list[str], header_token: str) -> pd.DataFrame | None:
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and header_token in line:
            if i + 2 >= len(lines):
                return None
            header = parse_md_row(line)
            rows = []
            for j in range(i + 2, len(lines)):
                if not lines[j].strip().startswith("|"):
                    break
                rows.append(parse_md_row(lines[j]))
            if not rows:
                return None
            return pd.DataFrame(rows, columns=header)
    return None


def load_supervised_metrics(report_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    lines = read_lines_safe(report_path)
    if not lines:
        return None, None

    class_df = parse_markdown_table(lines, "CV_AUC")
    reg_df = parse_markdown_table(lines, "CV_MAE")

    if class_df is not None:
        class_df = class_df.rename(
            columns={
                "Modelo": "Model",
                "CV_AUC": "CV_AUC_mean",
                "Val_Acc": "Val_Accuracy",
                "Val_F1": "Val_F1",
                "Val_AUC": "Val_ROC_AUC",
                "Test_Acc": "Test_Accuracy",
                "Test_F1": "Test_F1",
                "Test_AUC": "Test_ROC_AUC",
            }
        )
        for col in class_df.columns:
            if col != "Model":
                class_df[col] = pd.to_numeric(class_df[col], errors="coerce")
        class_df = class_df.round(3)

    if reg_df is not None:
        reg_df = reg_df.rename(
            columns={
                "Modelo": "Model",
                "Val_MAE": "Val_MAE",
                "Val_RMSE": "Val_RMSE",
                "Val_R2": "Val_R2",
                "Test_MAE": "Test_MAE",
                "Test_RMSE": "Test_RMSE",
                "Test_R2": "Test_R2",
            }
        )
        for col in reg_df.columns:
            if col != "Model":
                reg_df[col] = pd.to_numeric(reg_df[col], errors="coerce")
        reg_df = reg_df.round(3)

    return class_df, reg_df


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


# Cargar datasets
DF_FEATURES = read_csv_safe(FEATURES_PATH)
DF_CLUSTERS = read_csv_safe(CLUSTERS_PATH)
DF_CRYPTO = read_csv_safe(CRYPTO_RAW)
DF_CLASS, DF_REG = load_supervised_metrics(SUPERVISED_REPORT)


# -----------------------------------------------------------------------------
# Preparacion de datos
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
ANOMALY_COLUMNS: list[str] = []
DF_CLUSTER = None

if DF_FEATURES is not None and DF_CLUSTERS is not None:
    if "Symbol" in DF_FEATURES.columns and "Symbol" in DF_CLUSTERS.columns:
        CLUSTER_COLUMNS = [c for c in DF_CLUSTERS.columns if c != "Symbol"]
        ANOMALY_COLUMNS = [c for c in CLUSTER_COLUMNS if "OneClass" in c]
        CLUSTER_COLUMNS = [c for c in CLUSTER_COLUMNS if c not in ANOMALY_COLUMNS]
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

            rows = []
            for col in CLUSTER_COLUMNS:
                labels = merged[col].values
                valid_mask = labels != -1
                unique_labels = sorted(set(labels[valid_mask]))
                n_clusters = len(unique_labels)
                row = {
                    "Model": col,
                    "N_Clusters": float(n_clusters),
                }
                if n_clusters >= 2 and valid_mask.sum() >= 2:
                    try:
                        row["Silhouette"] = silhouette_score(scaled[valid_mask], labels[valid_mask])
                        row["Davies_Bouldin"] = davies_bouldin_score(scaled[valid_mask], labels[valid_mask])
                        row["Calinski_Harabasz"] = calinski_harabasz_score(
                            scaled[valid_mask], labels[valid_mask]
                        )
                    except Exception:
                        row["Silhouette"] = None
                        row["Davies_Bouldin"] = None
                        row["Calinski_Harabasz"] = None
                else:
                    row["Silhouette"] = None
                    row["Davies_Bouldin"] = None
                    row["Calinski_Harabasz"] = None
                rows.append(row)
            DF_CLUSTER = pd.DataFrame(rows)
            DF_CLUSTER = DF_CLUSTER.round(3)


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

TABLE_LABELS = {
    "Model": "Modelo",
}


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
    return f"{value:.3f}"


# -----------------------------------------------------------------------------
# Aplicacion Dash
# -----------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
app = Dash(
    __name__,
    title="Crypto ML Dashboard",
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
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
                html.Div("Mejor clasificador", className="kpi-title"),
                html.Div(best_clf[0] if best_clf else "n/a", className="kpi-value"),
                html.Div(f"Test ROC AUC: {format_kpi(best_clf[1] if best_clf else None)}", className="kpi-meta"),
            ],
        ),
        html.Div(
            className="kpi-card",
            children=[
                html.Div("Mejor regresor", className="kpi-title"),
                html.Div(best_reg[0] if best_reg else "n/a", className="kpi-value"),
                html.Div(f"Test R2: {format_kpi(best_reg[1] if best_reg else None)}", className="kpi-meta"),
            ],
        ),
        html.Div(
            className="kpi-card",
            children=[
                html.Div("Mejor clustering", className="kpi-title"),
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
                        html.H1("Dashboard - Análisis Cripto con ML"),
                        html.P("Comparacion de modelos y metricas de rendimiento"),
                    ],
                ),
                html.Div(
                    className="hero-note",
                    children="Cortisol Team.",
                ),
            ],
        ),
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
                                        html.H3("Análisis exploratorio de datos"),
                                        html.P("Vistas interactivas de precios, retornos y volatilidad."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Simbolo"),
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
                                                html.Label("Rango de fechas"),
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
                                                html.Label("Etiqueta de cluster"),
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
                                                html.Label("Metrica"),
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
                                    columns=[
                                        {"name": TABLE_LABELS.get(c, c), "id": c}
                                        for c in (CLUSTER_TABLE_COLS if DF_CLUSTER is not None else [])
                                    ],
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
                                        html.H3("Detección de anomalías - OneClass SVM"),
                                        html.P("Anomalías resaltadas en rojo y separadas del resto."),
                                    ],
                                ),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div(
                                            className="control",
                                            children=[
                                                html.Label("Algoritmo OneClass SVM"),
                                                dcc.Dropdown(
                                                    id="anomaly-algo",
                                                    options=[{"label": c, "value": c} for c in ANOMALY_COLUMNS],
                                                    value=(ANOMALY_COLUMNS[0] if ANOMALY_COLUMNS else None),
                                                    clearable=False,
                                                    disabled=not ANOMALY_COLUMNS,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="graph-grid",
                                    children=[
                                        dcc.Graph(id="anomaly-pca"),
                                        dcc.Graph(id="anomaly-stats"),
                                    ],
                                ),
                                dash_table.DataTable(
                                    id="anomaly-table",
                                    columns=[
                                        {"name": "Moneda", "id": "Symbol"},
                                        {"name": "Tipo", "id": "Type"},
                                    ],
                                    data=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_header={"backgroundColor": "#0f172a", "color": "white"},
                                    style_cell={"padding": "8px", "fontFamily": "Space Grotesk"},
                                ),
                            ],
                        ),
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
                                                html.Label("Metrica"),
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
                                    columns=[
                                        {"name": TABLE_LABELS.get(c, c), "id": c}
                                        for c in (CLASS_TABLE_COLS if DF_CLASS is not None else [])
                                    ],
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
                                                html.Label("Metrica"),
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
                                    columns=[
                                        {"name": TABLE_LABELS.get(c, c), "id": c}
                                        for c in (REG_TABLE_COLS if DF_REG is not None else [])
                                    ],
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
    Input("eda-symbol", "value"),
    Input("eda-date-range", "start_date"),
    Input("eda-date-range", "end_date"),
)
def update_eda(symbol: str | None, start_date: str | None, end_date: str | None):
    if not CRYPTO_AVAILABLE or symbol is None:
        message = "crypto_raw.csv no disponible"
        return (
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
        message = "Sin datos para el rango seleccionado"
        return (
            empty_figure(message),
            empty_figure(message),
            empty_figure(message),
        )

    fig_price = px.line(df, x="Date", y="Close", title="Precio de cierre")
    fig_price.update_layout(margin=dict(l=30, r=20, t=50, b=30))
    fig_price.update_yaxes(tickformat=".3f", hoverformat=".3f")

    df_returns = df.dropna(subset=["Return"]).copy()
    if df_returns.empty:
        fig_returns = empty_figure("No hay suficientes datos de retornos")
    else:
        fig_returns = px.histogram(df_returns, x="Return", nbins=60, title="Distribucion de retornos")
        fig_returns.update_layout(margin=dict(l=30, r=20, t=50, b=30))
        fig_returns.update_xaxes(tickformat=".3f", hoverformat=".3f")
        fig_returns.update_yaxes(tickformat=".3f", hoverformat=".3f")

    df_vol = df.dropna(subset=["RollingVol30", "RollingVol90"], how="all")
    if df_vol.empty:
        fig_vol = empty_figure("No hay suficientes datos de volatilidad")
    else:
        fig_vol = px.line(
            df_vol,
            x="Date",
            y=["RollingVol30", "RollingVol90"],
            title="Volatilidad rodante (30d / 90d)",
        )
        fig_vol.update_layout(margin=dict(l=30, r=20, t=50, b=30))
        fig_vol.update_yaxes(tickformat=".3f", hoverformat=".3f")

    return fig_price, fig_returns, fig_vol


@app.callback(
    Output("cluster-pca", "figure"),
    Input("cluster-algo", "value"),
)
def update_cluster_pca(cluster_col: str | None):
    if PCA_DATA is None or not cluster_col or cluster_col not in PCA_DATA.columns:
        return empty_figure("Datos de clustering no disponibles")

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
        title=f"Proyeccion PCA ({cluster_col})",
    )
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
    fig.update_xaxes(tickformat=".3f", hoverformat=".3f")
    fig.update_yaxes(tickformat=".3f", hoverformat=".3f")
    return fig


@app.callback(
    Output("cluster-metrics", "figure"),
    Input("cluster-metric", "value"),
)
def update_cluster_metrics(metric: str | None):
    if DF_CLUSTER is None or DF_CLUSTER.empty or metric not in DF_CLUSTER.columns:
        return empty_figure("Metricas de clustering no disponibles")

    df = DF_CLUSTER[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("La metrica no tiene valores")

    fig = px.bar(df, x="Model", y=metric, title="Comparacion de metricas de clustering")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    fig.update_yaxes(tickformat=".3f", hoverformat=".3f")
    return fig


@app.callback(
    Output("clf-metric-graph", "figure"),
    Input("clf-metric", "value"),
)
def update_clf_metric(metric: str | None):
    if DF_CLASS is None or DF_CLASS.empty or metric not in DF_CLASS.columns:
        return empty_figure("Metricas de clasificacion no disponibles")

    df = DF_CLASS[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("La metrica no tiene valores")

    fig = px.bar(df, x="Model", y=metric, title="Comparacion de clasificadores")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    fig.update_yaxes(tickformat=".3f", hoverformat=".3f")
    return fig


@app.callback(
    Output("reg-metric-graph", "figure"),
    Input("reg-metric", "value"),
)
def update_reg_metric(metric: str | None):
    if DF_REG is None or DF_REG.empty or metric not in DF_REG.columns:
        return empty_figure("Metricas de regresion no disponibles")

    df = DF_REG[["Model", metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty_figure("La metrica no tiene valores")

    fig = px.bar(df, x="Model", y=metric, title="Comparacion de regresores")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    fig.update_yaxes(tickformat=".3f", hoverformat=".3f")
    return fig


@app.callback(
    Output("anomaly-pca", "figure"),
    Output("anomaly-stats", "figure"),
    Output("anomaly-table", "data"),
    Input("anomaly-algo", "value"),
)
def update_anomaly_detection(anomaly_col: str | None):
    if PCA_DATA is None or not anomaly_col or anomaly_col not in PCA_DATA.columns:
        return (
            empty_figure("Datos de anomalias no disponibles"),
            empty_figure("Datos de anomalias no disponibles"),
            [],
        )

    df_plot = PCA_DATA.copy()
    df_plot["AnomalyLabel"] = df_plot[anomaly_col].astype(str)
    
    # Separar anomalías y normales
    anomalies = df_plot[df_plot["AnomalyLabel"] == "-1"].copy()
    normals = df_plot[df_plot["AnomalyLabel"] != "-1"].copy()
    
    # Grafico PCA con anomalias destacadas
    fig_pca = go.Figure()
    
    # Graficar puntos normales
    if len(normals) > 0:
        fig_pca.add_trace(go.Scatter(
            x=normals["PC1"],
            y=normals["PC2"],
            mode="markers",
            name="Normal",
            marker=dict(size=8, color="#3b82f6", opacity=0.6),
            text=normals["Symbol"],
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        ))
    
    # Graficar anomalías en rojo
    if len(anomalies) > 0:
        fig_pca.add_trace(go.Scatter(
            x=anomalies["PC1"],
            y=anomalies["PC2"],
            mode="markers",
            name="Anomalia",
            marker=dict(size=10, color="#ef4444", symbol="diamond", opacity=0.9),
            text=anomalies["Symbol"],
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        ))
    
    fig_pca.update_layout(
        title=f"Proyeccion PCA - Anomalias ({anomaly_col})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="closest",
        legend=dict(x=0.01, y=0.99),
    )
    fig_pca.update_xaxes(tickformat=".3f")
    fig_pca.update_yaxes(tickformat=".3f")
    
    # Grafico de estadisticas
    n_total = len(df_plot)
    n_anomalies = len(anomalies)
    n_normal = len(normals)
    pct_anomalies = (n_anomalies / n_total * 100) if n_total > 0 else 0
    
    stats_data = pd.DataFrame({
        "Tipo": ["Normal", "Anomalia"],
        "Cantidad": [n_normal, n_anomalies],
    })
    
    fig_stats = px.bar(
        stats_data,
        x="Tipo",
        y="Cantidad",
        color="Tipo",
        color_discrete_map={"Normal": "#3b82f6", "Anomalia": "#ef4444"},
        title=f"Distribucion de puntos (Total: {n_total} | Anomalias: {pct_anomalies:.1f}%)",
    )
    fig_stats.update_layout(
        margin=dict(l=30, r=20, t=50, b=30),
        showlegend=False,
    )
    fig_stats.update_yaxes(tickformat=".3f", hoverformat=".3f")
    
    # Tabla de anomalias
    if len(anomalies) > 0:
        table_data = [
            {"Symbol": row["Symbol"], "Type": "Anomalia"}
            for _, row in anomalies.iterrows()
        ]
    else:
        table_data = []
    
    return fig_pca, fig_stats, table_data


if __name__ == "__main__":
    app.run(debug=True)

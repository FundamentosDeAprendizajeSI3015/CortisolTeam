"""
Dashboard Interactivo HTML — Crypto ML Project
Genera un dashboard HTML interactivo para visualizar todas las graficas.

Uso:
    python visualization/app/generate_dashboard.py

Salida:
    visualization/app/dashboard.html — pagina interactiva con todas las visualizaciones
"""

from pathlib import Path
import sys

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ──────────────────────────────────────────────────────────────────────────────

# Ruta de salida
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"

# Informacion de las visualizaciones
VISUALIZACIONES = [
    {
        'titulo': 'Series de Tiempo de Precios',
        'archivo': '01_price_timeseries.png',
        'descripcion': 'Evolucion historica de los precios de cierre para BTC, ETH y BNB.'
    },
    {
        'titulo': 'Precios Normalizados',
        'archivo': '02_price_normalized.png',
        'descripcion': 'Comparacion del crecimiento relativo de las monedas (base = 100).'
    },
    {
        'titulo': 'Distribucion de Retornos',
        'archivo': '03_returns_distribution.png',
        'descripcion': 'Histogramas de retornos diarios con distribucion normal superpuesta.'
    },
    {
        'titulo': 'Correlacion de Retornos',
        'archivo': '04_correlation_heatmap.png',
        'descripcion': 'Matriz de correlacion de Pearson entre retornos diarios de todas las monedas.'
    },
    {
        'titulo': 'Analisis de Volumen',
        'archivo': '05_volume_analysis.png',
        'descripcion': 'Volumen de trading en escala logaritmica para monedas principales.'
    },
    {
        'titulo': 'Rolling Volatility',
        'archivo': '09_rolling_volatility.png',
        'descripcion': 'Evolucion temporal de volatilidad rodante a 30 y 90 dias para BTC, ETH y BNB.'
    },
    {
        'titulo': 'Seasonality Mensual',
        'archivo': '10_seasonality_month.png',
        'descripcion': 'Retorno promedio por mes para identificar patrones estacionales en cada moneda.'
    },
    {
        'titulo': 'Seasonality por Dia de Semana',
        'archivo': '11_seasonality_dayofweek.png',
        'descripcion': 'Retorno promedio por dia de semana para comparar efectos temporales recurrentes.'
    },
    {
        'titulo': 'Deteccion de Outliers',
        'archivo': '12_outliers_btc_returns.png',
        'descripcion': 'Outliers en retornos diarios de BTC detectados con Z-Score e IQR.'
    },
    {
        'titulo': 'Seleccion de K',
        'archivo': '13_k_selection_metrics.png',
        'descripcion': 'Comparacion de Metodo del Codo, Silhouette y Davies-Bouldin para elegir K.'
    },
    {
        'titulo': 'Visualizacion PCA — Clustering',
        'archivo': '06_pca_clustering.png',
        'descripcion': 'Proyeccion 2D de clusters en espacio PCA (diferentes algoritmos).'
    },
    {
        'titulo': 'Heatmap de Features',
        'archivo': '07_features_heatmap.png',
        'descripcion': 'Features normalizadas por moneda, coloreadas por cluster K4.'
    },
    {
        'titulo': 'Estadisticas por Cluster',
        'archivo': '08_cluster_statistics.png',
        'descripcion': 'Distribucion de features por cluster usando boxplots.'
    }
]


# ──────────────────────────────────────────────────────────────────────────────
# PLANTILLA HTML
# ──────────────────────────────────────────────────────────────────────────────

def generar_html() -> str:
    """
    Genera el codigo HTML del dashboard interactivo.
    
    Returns:
        String con el codigo HTML completo
    """
    
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Interactivo — Crypto ML Project</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-1: #f9f5ef;
            --bg-2: #e6f6ff;
            --accent: #0ea5a4;
            --accent-dark: #0f766e;
            --ink: #1f2937;
            --muted: #55636f;
            --card: #ffffff;
            --border: #e5e7eb;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background:
                radial-gradient(900px 600px at -10% -20%, #fff1da 0%, transparent 60%),
                radial-gradient(800px 520px at 110% 0%, #dff6ff 0%, transparent 55%),
                linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 100%);
            min-height: 100vh;
            padding: 20px;
            color: var(--ink);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: var(--card);
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #0f766e 0%, #0ea5a4 45%, #38bdf8 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-family: 'Fraunces', serif;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.95;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            font-size: 0.95em;
            opacity: 0.85;
        }
        
        .nav-tabs {
            display: flex;
            flex-wrap: wrap;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
            padding: 0;
            margin: 0;
        }
        
        .nav-tabs button {
            flex: 1;
            padding: 15px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 500;
            color: #555;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        
        .nav-tabs button:hover {
            background: #f0f0f0;
            color: var(--accent-dark);
        }
        
        .nav-tabs button.active {
            color: var(--accent-dark);
            border-bottom-color: var(--accent);
            background: #ecfeff;
        }
        
        .content {
            padding: 40px;
            display: grid;
            gap: 40px;
        }
        
        .section {
            display: none;
        }
        
        .section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .section-title {
            font-size: 1.8em;
            color: var(--ink);
            margin-bottom: 10px;
            font-weight: 700;
            padding-bottom: 10px;
            border-bottom: 3px solid var(--accent);
        }

        .section-intro {
            color: var(--muted);
            font-size: 1.05em;
            margin: 8px 0 24px;
        }
        
        .visualization {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 30px;
        }
        
        .visualization:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        .visualization-header {
            background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 100%);
            color: white;
            padding: 20px;
        }
        
        .visualization-header h3 {
            font-size: 1.3em;
            margin-bottom: 5px;
        }
        
        .visualization-header p {
            font-size: 0.95em;
            opacity: 0.9;
            margin: 0;
        }
        
        .visualization-body {
            padding: 0;
            overflow: auto;
        }
        
        .visualization-body img {
            width: 100%;
            height: auto;
            display: block;
        }

        .placeholder-box {
            background: #f7f9fc;
            border: 2px dashed #9fb3d9;
            border-radius: 10px;
            padding: 26px;
            color: #2c3e63;
            font-size: 1.05em;
            line-height: 1.6;
        }

        .note-card {
            background: #f3fbfb;
            border: 1px solid #cde8e6;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 26px;
            color: #1f3b3a;
            font-size: 1.02em;
        }

        .note-card a {
            color: var(--accent-dark);
            text-decoration: none;
            font-weight: 600;
        }

        .note-card a:hover {
            text-decoration: underline;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-top: 2px solid #e0e0e0;
            color: #666;
            font-size: 0.9em;
        }
        
        .footer a {
            color: var(--accent-dark);
            text-decoration: none;
            font-weight: 600;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .nav-tabs button {
                font-size: 0.85em;
                padding: 12px 15px;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <div class="header">
            <h1>Dashboard Interactivo</h1>
            <p>Crypto ML Project — Analisis de Criptomonedas</p>
            <div class="subtitle">Visualizaciones de EDA, Modelo No Supervisado, Modelo Supervisado y Scoring</div>
        </div>
        
        <!-- TABS DE NAVEGACION -->
        <div class="nav-tabs">
            <button class="tab-btn active" onclick="mostrarSeccion(0)">
                Analisis Exploratorio (EDA)
            </button>
            <button class="tab-btn" onclick="mostrarSeccion(1)">
                Modelo No Supervisado
            </button>
            <button class="tab-btn" onclick="mostrarSeccion(2)">
                Modelo Supervisado
            </button>
            <button class="tab-btn" onclick="mostrarSeccion(3)">
                Scoring
            </button>
        </div>
        
        <!-- CONTENIDO -->
        <div class="content">
            <!-- SECCION 1: EDA -->
            <div class="section active" id="seccion-0">
                <h2 class="section-title">Analisis Exploratorio de Datos (EDA)</h2>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Series de Tiempo de Precios</h3>
                        <p>Evolucion historica de los precios de cierre para BTC, ETH y BNB en USD</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/01_price_timeseries.png" alt="Series de Tiempo">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Precios Normalizados</h3>
                        <p>Comparacion del crecimiento relativo de las monedas (base = 100 al inicio)</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/02_price_normalized.png" alt="Precios Normalizados">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Distribucion de Retornos Diarios</h3>
                        <p>Histogramas de retornos con distribucion normal superpuesta para identificar patrones</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/03_returns_distribution.png" alt="Distribucion de Retornos">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Matriz de Correlacion</h3>
                        <p>Correlacion de Pearson entre retornos diarios de todas las criptomonedas</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/04_correlation_heatmap.png" alt="Correlacion">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Analisis de Volumen de Trading</h3>
                        <p>Volumen de transacciones en escala logaritmica para capturar variaciones amplias</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/05_volume_analysis.png" alt="Volumen">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Rolling Volatility (30d y 90d)</h3>
                        <p>Evolucion temporal de la volatilidad para identificar periodos de mayor riesgo</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/09_rolling_volatility.png" alt="Rolling Volatility">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Seasonality Mensual</h3>
                        <p>Retorno promedio por mes para detectar estacionalidad en BTC, ETH y BNB</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/10_seasonality_month.png" alt="Seasonality Mensual">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Seasonality por Dia de Semana</h3>
                        <p>Retorno promedio por dia para evaluar patrones semanales de mercado</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/11_seasonality_dayofweek.png" alt="Seasonality Dia Semana">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Deteccion de Outliers en Retornos BTC</h3>
                        <p>Eventos extremos detectados con metodos Z-Score e IQR para analizar shocks de mercado</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/12_outliers_btc_returns.png" alt="Outliers BTC">
                    </div>
                </div>
            </div>
            
            <!-- SECCION 2: NO SUPERVISADO -->
            <div class="section" id="seccion-1">
                <h2 class="section-title">Resultados del Modelo No Supervisado</h2>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Seleccion de K</h3>
                        <p>Comparacion de codo, silhouette y Davies-Bouldin para soportar la eleccion del K optimo</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/13_k_selection_metrics.png" alt="K Selection Metrics">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Visualizacion en Espacio PCA</h3>
                        <p>Proyeccion 2D de clusters usando PCA (K-Means K2, K4, DBSCAN, Agglomerative)</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/06_pca_clustering.png" alt="PCA Clustering">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Heatmap de Features por Moneda</h3>
                        <p>Features normalizadas organizadas por cluster K4 (volatility, sharpe ratio, etc)</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/07_features_heatmap.png" alt="Features Heatmap">
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Estadisticas por Cluster</h3>
                        <p>Distribucion de features para cada cluster usando boxplots (media, mediana, rango)</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../outputs/08_cluster_statistics.png" alt="Cluster Statistics">
                    </div>
                </div>
            </div>

            <!-- SECCION 3: SUPERVISADO -->
            <div class="section" id="seccion-2">
                <h2 class="section-title">Resultados del Modelo Supervisado</h2>

                <p class="section-intro">Resumen de clasificacion y regresion con curvas de aprendizaje y diagnosticos clave.</p>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Tabla Comparativa de Metricas</h3>
                        <p>Resumen consolidado de resultados en validacion y test.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/28_tabla_metricas.png" alt="Tabla de metricas">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Comparacion de Clasificadores</h3>
                        <p>ROC-AUC en validacion y test para Logistic, RandomForest y XGBoost.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/13_comparacion_clasificadores.png" alt="Comparacion clasificadores">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Comparacion de Regresores</h3>
                        <p>R2 en validacion y test para Ridge, RandomForest y XGBoost.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/14_comparacion_regresores.png" alt="Comparacion regresores">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Matriz de Confusion (Clasificador)</h3>
                        <p>Desempeno del mejor clasificador en el conjunto de test.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/15_confusion_matrix.png" alt="Matriz de confusion">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Curva ROC (Clasificador)</h3>
                        <p>Comparacion de sensibilidad vs especificidad en test.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/16_roc_curve.png" alt="Curva ROC">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Curva de Aprendizaje — Logistic Regression</h3>
                        <p>Evolucion del rendimiento del clasificador con mas datos.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/26_curva_aprendizaje_log.png" alt="Curva aprendizaje clasificador">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Curva de Aprendizaje — XGBRegressor</h3>
                        <p>Impacto del tamano de entrenamiento en el regresor.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/27_curva_aprendizaje_xgb.png" alt="Curva aprendizaje regresor">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Importancia de Features — Clasificador</h3>
                        <p>Ranking de variables para explicar sube/baja.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/17_feature_importance_ran.png" alt="Importancia de features clasificador">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Importancia de Features — Regresor</h3>
                        <p>Variables con mayor impacto en la prediccion de retorno.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/18_feature_importance_xgb.png" alt="Importancia de features regresor">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Predicciones vs Real</h3>
                        <p>Comparacion directa entre retorno real y estimado.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/19_predicciones_vs_real.png" alt="Predicciones vs real">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Distribucion de Residuos</h3>
                        <p>Diagnostico de errores del mejor regresor.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/20_residuos.png" alt="Distribucion de residuos">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Scatter Matrix de Features</h3>
                        <p>Relacion entre variables clave y la clase objetivo.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/29_scatter_matrix.png" alt="Scatter matrix">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Heatmap de Correlacion (Features + Lags)</h3>
                        <p>Dependencias entre indicadores tecnicos y lags temporales.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/24_heatmap_correlacion.png" alt="Heatmap correlacion">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Distribucion de Return por Simbolo</h3>
                        <p>Boxplot de retornos diarios sin outliers extremos.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/25_boxplot_return_por_simbolo.png" alt="Boxplot return por simbolo">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>3D: Features vs Return</h3>
                        <p>Relaciones no lineales entre indicadores y retorno.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/21_3d_features_vs_return.png" alt="3D features vs return">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>3D: Real vs Predicho vs Residuo</h3>
                        <p>Visualizacion 3D de error del regresor.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../supervised/reports/figures/22_3d_predicciones.png" alt="3D real vs predicho">
                    </div>
                </div>
            </div>

            <!-- SECCION 4: SCORING -->
            <div class="section" id="seccion-3">
                <h2 class="section-title">Scoring y Resultados Finales</h2>

                <p class="section-intro">Metricas estandarizadas, comparaciones finales y backtesting.</p>

                <div class="note-card">
                    <strong>Archivos clave:</strong>
                    <a href="../../scoring/reports/metrics_summary.csv" target="_blank" rel="noopener">metrics_summary.csv</a>
                    &middot;
                    <a href="../../scoring/reports/model_report.pdf" target="_blank" rel="noopener">model_report.pdf</a>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Comparacion de Clasificacion</h3>
                        <p>Metricas principales para todos los clasificadores en test.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/17_scoring_clf_comparison.png" alt="Scoring clasificacion">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Matrices de Confusion</h3>
                        <p>Resumen visual del desempeno por clase.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/18_scoring_confusion_matrices.png" alt="Scoring matrices confusion">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>ROC Curves</h3>
                        <p>Comparacion de tradeoff TPR/FPR entre modelos.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/19_scoring_roc_curves.png" alt="Scoring ROC curves">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Regresion: Predicho vs Real</h3>
                        <p>Dispersion de predicciones y linea de referencia.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/20_scoring_regression_scatter.png" alt="Scoring regresion scatter">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Regresion: Residuos</h3>
                        <p>Distribucion de errores para validar supuestos.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/21_scoring_residuals.png" alt="Scoring residuos">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Metricas de Clustering</h3>
                        <p>Silhouette, Davies-Bouldin y Calinski-Harabasz.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/22_scoring_clustering_metrics.png" alt="Scoring clustering metrics">
                    </div>
                </div>

                <div class="visualization">
                    <div class="visualization-header">
                        <h3>Backtesting</h3>
                        <p>Comparacion de estrategia del modelo vs buy & hold.</p>
                    </div>
                    <div class="visualization-body">
                        <img src="../../scoring/reports/figures/23_scoring_backtest.png" alt="Scoring backtesting">
                    </div>
                </div>
            </div>
        </div>
        
        <!-- FOOTER -->
        <div class="footer">
            <p>Dashboard generado automaticamente por <strong>Crypto ML Project</strong></p>
            <p>Ultima actualizacion: <span id="fecha"></span></p>
        </div>
    </div>
    
    <script>
        // Funcion para cambiar de seccion
        function mostrarSeccion(indice) {
            // Ocultar todas las secciones
            const secciones = document.querySelectorAll('.section');
            secciones.forEach(s => s.classList.remove('active'));
            
            // Desactivar todos los botones
            const botones = document.querySelectorAll('.tab-btn');
            botones.forEach(b => b.classList.remove('active'));
            
            // Mostrar seccion seleccionada
            document.getElementById(`seccion-${indice}`).classList.add('active');
            botones[indice].classList.add('active');
            
            // Scroll al inicio
            window.scrollTo(0, 0);
        }
        
        // Establecer fecha actual
        document.getElementById('fecha').textContent = new Date().toLocaleString('es-ES');
    </script>
</body>
</html>"""
    
    return html


# ──────────────────────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Funcion principal que genera el dashboard HTML.
    """
    print("\n" + "="*70)
    print(" GENERADOR DE DASHBOARD HTML — Crypto ML Project")
    print("="*70 + "\n")
    
    # Validar que exista el directorio de outputs
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] No se encontro el directorio: {OUTPUT_DIR}")
        print("[INFO] Ejecuta primero: python visualization/app/visualize.py")
        sys.exit(1)
    
    # Validar que existan todas las imagenes
    print("[VALIDACION] Verificando visualizaciones...")
    imagenes_faltantes = []
    
    for viz in VISUALIZACIONES:
        img_path = OUTPUT_DIR / viz['archivo']
        if not img_path.exists():
            imagenes_faltantes.append(viz['archivo'])
            print(f"[AVISO] No encontrado: {viz['archivo']}")
        else:
            print(f"[OK] Encontrado: {viz['archivo']}")
    
    if imagenes_faltantes:
        print(f"\n[ERROR] Faltan {len(imagenes_faltantes)} visualizaciones.")
        print("[INFO] Ejecuta primero: python visualization/app/visualize.py")
        sys.exit(1)
    
    # Generar HTML
    print("\n[GENERACION] Creando dashboard HTML...")
    html_content = generar_html()
    
    # Guardar archivo
    try:
        with open(DASHBOARD_PATH, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[OK] Dashboard guardado: {DASHBOARD_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el dashboard: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" COMPLETADO: Dashboard generado exitosamente.")
    print(f" Ubicacion: {DASHBOARD_PATH}")
    print(" Abre el archivo en tu navegador para visualizarlo.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

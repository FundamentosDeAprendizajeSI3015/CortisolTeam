"""
Visualizacion Interactiva — Crypto ML Project
Genera y guarda visualizaciones a partir de datos EDA y clustering.

Uso:
    python visualization/app/visualize.py

Salida:
    visualization/outputs/  — directorio con todas las figuras (PNG)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ──────────────────────────────────────────────────────────────────────────────

# Raiz del proyecto
ROOT = Path(__file__).resolve().parents[2]

# Rutas de datos de entrada
DATA_DIR = ROOT / "data"
CSV_CRYPTO = DATA_DIR / "crypto_raw.csv"
CSV_FEATURES = DATA_DIR / "features_clustering.csv"
CSV_CLUSTERS = DATA_DIR / "cluster_labels.csv"

# Ruta de salida para las visualizaciones
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colores consistentes
COLORES = {
    'BTC': '#F7931A',
    'ETH': '#627EEA',
    'BNB': '#F3BA2F',
    'ADA': '#0033A0',
    'SOL': '#14F195',
    'DOGE': '#BA9F33'
}

# Monedas principales para enfoque
MONEDAS_PRINCIPALES = ['BTC', 'ETH', 'BNB']
MESES_ES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
DIAS_ES = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']

# Configuracion de estilos Seaborn y Matplotlib
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE VALIDACION
# ──────────────────────────────────────────────────────────────────────────────

def validar_archivo(path: Path, nombre: str) -> bool:
    """
    Valida que un archivo exista.
    
    Args:
        path: Ruta del archivo
        nombre: Nombre descriptivo para el mensaje de error
        
    Returns:
        True si el archivo existe, False en caso contrario
    """
    if not path.exists():
        print(f"[ERROR] No se encontro {nombre} en: {path}")
        return False
    print(f"[OK] Cargado: {nombre}")
    return True


def cargar_datos(path: Path) -> pd.DataFrame:
    """
    Carga un archivo CSV de forma segura.
    
    Args:
        path: Ruta del archivo CSV
        
    Returns:
        DataFrame con los datos
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {path}: {e}")
        return None


def validar_columnas(df: pd.DataFrame, columnas_requeridas: list, 
                     nombre_df: str) -> bool:
    """
    Valida que un DataFrame tenga las columnas requeridas.
    
    Args:
        df: DataFrame a validar
        columnas_requeridas: Lista de nombres de columnas esperadas
        nombre_df: Nombre del DataFrame para mensajes de error
        
    Returns:
        True si todas las columnas existen, False en caso contrario
    """
    columnas_faltantes = set(columnas_requeridas) - set(df.columns)
    if columnas_faltantes:
        print(f"[ERROR] {nombre_df} carece de columnas: {columnas_faltantes}")
        return False
    return True


def preparar_retornos(df_crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara retornos y columnas de calendario para visualizaciones temporales.

    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close.

    Returns:
        DataFrame ordenado con Return, mes y dia_semana.
    """
    df_temp = df_crypto.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='mixed')
    df_temp = df_temp.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    df_temp['Return'] = df_temp.groupby('Symbol')['Close'].pct_change()
    df_temp['mes'] = df_temp['Date'].dt.month
    df_temp['dia_semana'] = df_temp['Date'].dt.dayofweek
    return df_temp


def detectar_outliers(serie: pd.Series, metodo: str = 'zscore',
                      umbral: float = 3.0) -> pd.Series:
    """
    Detecta outliers usando Z-Score o IQR.

    Args:
        serie: Serie numerica a analizar.
        metodo: Metodo de deteccion ('zscore' o 'iqr').
        umbral: Umbral para metodo zscore.

    Returns:
        Serie booleana con True en filas outlier.
    """
    s = serie.dropna()
    mask = pd.Series(False, index=serie.index)

    if s.empty:
        return mask

    if metodo == 'zscore':
        std = s.std()
        if pd.isna(std) or std == 0:
            return mask
        z_scores = (s - s.mean()) / std
        mask.loc[s.index] = np.abs(z_scores) > umbral
    elif metodo == 'iqr':
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return mask
        mask.loc[s.index] = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    else:
        raise ValueError(f"Metodo desconocido: {metodo}")

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE VISUALIZACION — EDA
# ──────────────────────────────────────────────────────────────────────────────

def plot_price_timeseries(df_crypto: pd.DataFrame) -> None:
    """
    Genera serie de tiempo de precios de cierre para monedas principales.
    Visualiza la evolucion de precios en USD a lo largo del tiempo.
    
    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return
    
    # Convertir Date a datetime
    df_crypto['Date'] = pd.to_datetime(df_crypto['Date'], format='mixed')
    
    # Filtrar monedas principales
    df_top = df_crypto[df_crypto['Symbol'].isin(MONEDAS_PRINCIPALES)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for sym in MONEDAS_PRINCIPALES:
        data = df_top[df_top['Symbol'] == sym]
        color = COLORES.get(sym, None)
        ax.plot(data['Date'], data['Close'], label=sym, linewidth=2, color=color)
    
    ax.set_title('Series de Tiempo de Precios de Cierre (BTC, ETH, BNB)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Precio (USD)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar figura
    output_path = OUTPUT_DIR / '01_price_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_price_normalized(df_crypto: pd.DataFrame) -> None:
    """
    Genera serie de tiempo de precios normalizados (base = 100).
    Permite comparar el crecimiento relativo de diferentes monedas.
    
    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return
    
    # Convertir Date a datetime
    df_crypto['Date'] = pd.to_datetime(df_crypto['Date'], format='mixed')
    
    # Filtrar monedas principales
    df_top = df_crypto[df_crypto['Symbol'].isin(MONEDAS_PRINCIPALES)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for sym in MONEDAS_PRINCIPALES:
        data = df_top[df_top['Symbol'] == sym].copy()
        # Normalizar: precio inicial = 100
        data['precio_norm'] = (data['Close'] / data['Close'].iloc[0]) * 100
        color = COLORES.get(sym, None)
        ax.plot(data['Date'], data['precio_norm'], label=sym, linewidth=2, color=color)
    
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Base (100)')
    ax.set_title('Precios Normalizados (Base = 100)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Precio Normalizado', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / '02_price_normalized.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_returns_distribution(df_crypto: pd.DataFrame) -> None:
    """
    Genera histogramas y distribuciones de retornos diarios para monedas principales.
    Visualiza la forma y caracteristicas de los retornos (simetria, colas).
    
    Args:
        df_crypto: DataFrame con columnas Symbol, Close
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Close'], 'crypto_raw'):
        return
    
    # Calcular retornos diarios
    df_temp = df_crypto.copy()
    df_temp['Return'] = df_temp.groupby('Symbol')['Close'].pct_change()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, sym in enumerate(MONEDAS_PRINCIPALES):
        data = df_temp[df_temp['Symbol'] == sym]['Return'].dropna()
        color = COLORES.get(sym, None)
        
        # Histograma con distribucion normal superpuesta
        axes[idx].hist(data, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
        
        # Curve gaussiana
        mu, sigma = data.mean(), data.std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        axes[idx].plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2), 
                      'r--', linewidth=2, label='Distribucion Normal')
        
        axes[idx].set_title(f'{sym} — Distribucion de Retornos Diarios', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Retorno (%)', fontsize=10)
        axes[idx].set_ylabel('Densidad', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / '03_returns_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_correlation_heatmap(df_crypto: pd.DataFrame) -> None:
    """
    Genera heatmap de correlacion entre retornos diarios de todas las monedas.
    Identifica monedas correlacionadas (movimientos similares).
    
    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return
    
    # Preparar datos
    df_temp = df_crypto.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='mixed')
    df_temp['Return'] = df_temp.groupby('Symbol')['Close'].pct_change()
    
    # Crear matriz pivote
    pivot = df_temp.pivot_table(index='Date', columns='Symbol', values='Return')
    
    # Calcular correlacion
    corr = pivot.corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Heatmap solo triangulo inferior (evitar redundancia)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlacion de Pearson'},
                annot_kws={'size': 7})
    
    ax.set_title('Matriz de Correlacion de Retornos Diarios', 
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / '04_correlation_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_volume_analysis(df_crypto: pd.DataFrame) -> None:
    """
    Genera visualizacion de volumen de trading en escala logaritmica.
    Muestra la actividad de trading para monedas principales.
    
    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Volume
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Volume'], 'crypto_raw'):
        return
    
    # Convertir Date a datetime
    df_crypto['Date'] = pd.to_datetime(df_crypto['Date'], format='mixed')
    
    # Filtrar monedas principales
    df_top = df_crypto[df_crypto['Symbol'].isin(MONEDAS_PRINCIPALES)].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, sym in enumerate(MONEDAS_PRINCIPALES):
        data = df_top[df_top['Symbol'] == sym]
        color = COLORES.get(sym, None)
        
        axes[idx].bar(data['Date'], data['Volume'], color=color, alpha=0.7, edgecolor='black')
        axes[idx].set_yscale('log')
        axes[idx].set_title(f'Volumen de Trading — {sym} (escala logaritmica)', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Fecha', fontsize=10)
        axes[idx].set_ylabel('Volumen (escala log)', fontsize=10)
        axes[idx].grid(True, alpha=0.3, which='both')
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / '05_volume_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_rolling_volatility(df_crypto: pd.DataFrame) -> None:
    """
    Genera volatilidad rodante a 30 y 90 dias para BTC, ETH y BNB.

    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close.
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return

    df_temp = preparar_retornos(df_crypto)
    df_temp['vol_30d'] = df_temp.groupby('Symbol')['Return'].transform(
        lambda x: x.rolling(30, min_periods=15).std())
    df_temp['vol_90d'] = df_temp.groupby('Symbol')['Return'].transform(
        lambda x: x.rolling(90, min_periods=45).std())

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for idx, sym in enumerate(MONEDAS_PRINCIPALES):
        data = df_temp[df_temp['Symbol'] == sym]
        axes[idx].plot(data['Date'], data['vol_30d'],
                       color='steelblue', linewidth=1.2, label='Std 30d')
        axes[idx].plot(data['Date'], data['vol_90d'],
                       color='tomato', linewidth=1.2, linestyle='--', label='Std 90d')
        axes[idx].set_title(f'Volatilidad Rodante - {sym}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Volatilidad', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx].legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Fecha', fontsize=10)
    plt.xticks(rotation=45)
    fig.suptitle('Rolling Volatility (Std de retornos diarios)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = OUTPUT_DIR / '09_rolling_volatility.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_seasonality_month(df_crypto: pd.DataFrame) -> None:
    """
    Genera estacionalidad mensual de retornos para BTC, ETH y BNB.

    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close.
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return

    df_temp = preparar_retornos(df_crypto)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, sym in enumerate(MONEDAS_PRINCIPALES):
        data = df_temp[df_temp['Symbol'] == sym].dropna(subset=['Return'])
        mes_avg = data.groupby('mes')['Return'].mean().reindex(range(1, 13))
        colores = ['tomato' if v < 0 else 'seagreen' for v in mes_avg.fillna(0)]

        axes[idx].bar(range(1, 13), mes_avg.values, color=colores)
        axes[idx].set_xticks(range(1, 13))
        axes[idx].set_xticklabels(MESES_ES, rotation=45)
        axes[idx].axhline(0, color='black', linewidth=0.8)
        axes[idx].set_title(f'{sym} - Retorno Promedio por Mes', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Mes', fontsize=10)
        axes[idx].set_ylabel('Retorno Promedio', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('Seasonality Mensual', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / '10_seasonality_month.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_seasonality_dayofweek(df_crypto: pd.DataFrame) -> None:
    """
    Genera estacionalidad por dia de semana de retornos para BTC, ETH y BNB.

    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close.
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return

    df_temp = preparar_retornos(df_crypto)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, sym in enumerate(MONEDAS_PRINCIPALES):
        data = df_temp[df_temp['Symbol'] == sym].dropna(subset=['Return'])
        dia_avg = data.groupby('dia_semana')['Return'].mean().reindex(range(7))
        colores = ['tomato' if v < 0 else 'steelblue' for v in dia_avg.fillna(0)]

        axes[idx].bar(range(7), dia_avg.values, color=colores)
        axes[idx].set_xticks(range(7))
        axes[idx].set_xticklabels(DIAS_ES, rotation=45)
        axes[idx].axhline(0, color='black', linewidth=0.8)
        axes[idx].set_title(f'{sym} - Retorno Promedio por Dia', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Dia de semana', fontsize=10)
        axes[idx].set_ylabel('Retorno Promedio', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('Seasonality por Dia de Semana', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / '11_seasonality_dayofweek.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_outliers_detection(df_crypto: pd.DataFrame) -> None:
    """
    Genera deteccion de outliers sobre retornos diarios de BTC.

    Args:
        df_crypto: DataFrame con columnas Symbol, Date, Close.
    """
    if not validar_columnas(df_crypto, ['Symbol', 'Date', 'Close'], 'crypto_raw'):
        return

    df_temp = preparar_retornos(df_crypto)
    btc = df_temp[df_temp['Symbol'] == 'BTC'].copy()

    btc['outlier_z'] = detectar_outliers(btc['Return'], metodo='zscore', umbral=3.0)
    btc['outlier_iqr'] = detectar_outliers(btc['Return'], metodo='iqr')

    btc_z = btc[btc['outlier_z']]
    btc_iqr_only = btc[btc['outlier_iqr'] & ~btc['outlier_z']]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(btc['Date'], btc['Return'], color='#F7931A', linewidth=1.2, alpha=0.85, label='BTC')
    ax.scatter(btc_z['Date'], btc_z['Return'], color='red', marker='x', s=45,
               linewidths=2, label='Outlier Z-Score')
    ax.scatter(btc_iqr_only['Date'], btc_iqr_only['Return'], color='orange', marker='^', s=35,
               label='Outlier IQR (solo IQR)')

    ax.set_title('BTC - Retornos Diarios con Outliers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Retorno Diario', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = OUTPUT_DIR / '12_outliers_btc_returns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE VISUALIZACION — CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def preparar_features_clustering(df_features: pd.DataFrame) -> tuple:
    """
    Prepara matriz de features normalizada y simbolos para visualizaciones.

    Args:
        df_features: DataFrame de features de clustering.

    Returns:
        Tupla con DataFrame indexado, matriz escalada y lista de simbolos.
    """
    if not validar_columnas(df_features, ['Symbol'], 'features_clustering'):
        return None, None, None

    df_pre = df_features.copy()
    if df_pre.index.name != 'Symbol' and 'Symbol' in df_pre.columns:
        df_pre = df_pre.set_index('Symbol')

    X = df_pre.select_dtypes(include=[np.number]).values
    symbols = df_pre.index.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df_pre, X_scaled, symbols


def plot_k_selection_metrics(df_features: pd.DataFrame) -> None:
    """
    Genera grafica de seleccion de K: codo, silhouette y Davies-Bouldin.

    Args:
        df_features: DataFrame con features de clustering.
    """
    _, X_scaled, _ = preparar_features_clustering(df_features)
    if X_scaled is None:
        return

    k_range = range(2, 8)
    inertias, silhouettes, db_scores = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(list(k_range), inertias, marker='o', color='tab:blue', linewidth=2)
    axes[0].set_title('Metodo del Codo', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('K', fontsize=10)
    axes[0].set_ylabel('Inercia', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, marker='o', color='green', linewidth=2)
    axes[1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('K', fontsize=10)
    axes[1].set_ylabel('Score', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(list(k_range), db_scores, marker='o', color='red', linewidth=2)
    axes[2].set_title('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('K', fontsize=10)
    axes[2].set_ylabel('Score (menor es mejor)', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / '13_k_selection_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")

def plot_pca_clustering(df_features: pd.DataFrame, df_clusters: pd.DataFrame) -> None:
    """
    Visualiza clusters en el espacio PCA 2D para diferentes algoritmos de clustering.
    Permite ver la separacion de grupos en los primeros dos componentes principales.
    
    Args:
        df_features: DataFrame con features para clustering
        df_clusters: DataFrame con etiquetas de clusters
    """
    if not validar_columnas(df_features, ['Symbol'], 'features_clustering'):
        return
    
    # Asegurarse de que Symbol es indice
    if df_features.index.name != 'Symbol':
        df_features = df_features.set_index('Symbol')
    
    # Features numericas (excluir Symbol si existe)
    X = df_features.select_dtypes(include=[np.number]).values
    symbols = df_features.index.tolist()
    
    # Estandarizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    
    # Varianza explicada
    var_explained = pca.explained_variance_ratio_.sum() * 100
    
    # Columnas de clustering disponibles (excluir Symbol si existe)
    cluster_cols = [col for col in df_clusters.columns if col != 'Symbol']
    
    # Crear subplots
    n_cols = 2
    n_rows = (len(cluster_cols) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten()
    
    # Mapeo de columnas de simbolos
    if df_clusters.index.name != 'Symbol':
        cluster_mapping = dict(zip(df_clusters['Symbol'].tolist(), range(len(df_clusters))))
    else:
        cluster_mapping = dict(zip(df_clusters.index.tolist(), range(len(df_clusters))))
    
    for ax_idx, col in enumerate(cluster_cols):
        ax = axes[ax_idx]
        
        # Obtener labels para esta columna
        if 'Symbol' in df_clusters.columns:
            labels = df_clusters.set_index('Symbol').loc[symbols, col].values
        else:
            labels = df_clusters.loc[symbols, col].values
        
        # Scatter plot con colores por cluster
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', 
                            s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Anotaciones con simbolos
        for i, sym in enumerate(symbols):
            ax.annotate(sym, (coords[i, 0], coords[i, 1]), 
                       fontsize=8, ha='center', va='center', fontweight='bold')
        
        ax.set_title(f'{col} (PCA — {var_explained:.1f}% varianza explicada)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=9)
    
    # Ocultar subplots no utilizados
    for ax_idx in range(len(cluster_cols), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / '06_pca_clustering.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_features_heatmap(df_features: pd.DataFrame, df_clusters: pd.DataFrame) -> None:
    """
    Visualiza heatmap de features normalizadas coloreadas por cluster K4.
    Permite identificar caracteristicas diferenciadoras entre clusters.
    
    Args:
        df_features: DataFrame con features para clustering
        df_clusters: DataFrame con etiquetas de clusters
    """
    if not validar_columnas(df_features, ['Symbol'], 'features_clustering'):
        return
    
    # Preparar datos
    df_features = df_features.copy()
    if df_features.index.name != 'Symbol' and 'Symbol' in df_features.columns:
        df_features = df_features.set_index('Symbol')
    
    if df_clusters.index.name != 'Symbol' and 'Symbol' in df_clusters.columns:
        df_clusters = df_clusters.set_index('Symbol')
    
    # Combinar features con labels K4
    df_plot = df_features.copy()
    
    # Usar la primera columna de clusters disponible si KMeans_K4 no existe
    cluster_col = 'KMeans_K4' if 'KMeans_K4' in df_clusters.columns else df_clusters.columns[0]
    df_plot['Cluster'] = df_clusters[cluster_col]
    
    # Ordenar por cluster
    df_plot = df_plot.sort_values('Cluster')
    
    # Features (excluir Cluster)
    X = df_plot.drop('Cluster', axis=1).values
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Heatmap
    im = ax.imshow(X.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    
    # Etiquetas
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(df_plot.columns)-1))
    ax.set_yticklabels(df_plot.columns[:-1], fontsize=10)
    
    # Titulo y colorbar
    ax.set_title('Heatmap de Features Normalizadas por Moneda', 
                fontsize=12, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor Normalizado (StandardScaler)', fontsize=10)
    
    # Lineas divisoras entre clusters
    unique_clusters = df_plot['Cluster'].unique()
    current_pos = 0
    for cluster in sorted(unique_clusters):
        cluster_size = (df_plot['Cluster'] == cluster).sum()
        ax.axvline(current_pos + cluster_size - 0.5, color='white', linewidth=2)
        current_pos += cluster_size
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / '07_features_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


def plot_cluster_statistics(df_features: pd.DataFrame, df_clusters: pd.DataFrame) -> None:
    """
    Genera graficas de estadisticas de features por cluster.
    Visualiza el perfil promedio de cada cluster usando boxplots.
    
    Args:
        df_features: DataFrame con features para clustering
        df_clusters: DataFrame con etiquetas de clusters
    """
    if not validar_columnas(df_features, ['Symbol'], 'features_clustering'):
        return
    
    # Preparar datos
    df_features = df_features.copy()
    if df_features.index.name != 'Symbol' and 'Symbol' in df_features.columns:
        df_features = df_features.set_index('Symbol')
    
    if df_clusters.index.name != 'Symbol' and 'Symbol' in df_clusters.columns:
        df_clusters = df_clusters.set_index('Symbol')
    
    # Usar K4 o primera columna disponible
    cluster_col = 'KMeans_K4' if 'KMeans_K4' in df_clusters.columns else df_clusters.columns[0]
    
    # Combinar
    df_combined = df_features.copy()
    df_combined['Cluster'] = df_clusters[cluster_col]
    
    # Features numericas (excluir Cluster)
    features = [col for col in df_combined.columns if col != 'Cluster']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:6]):
        ax = axes[idx]
        
        # Boxplot por cluster
        df_combined.boxplot(column=feature, by='Cluster', ax=ax)
        ax.set_title(f'{feature} por Cluster', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Valor Normalizado', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    # Ocultar subplots no utilizados
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Distribucion de Features por Cluster', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / '08_cluster_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {output_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Funcion principal que ejecuta todas las visualizaciones.
    """
    print("\n" + "="*70)
    print(" GENERADOR DE VISUALIZACIONES — Crypto ML Project")
    print("="*70 + "\n")
    
    # Validar que los archivos existan
    print("[VALIDACION] Verificando archivos de entrada...")
    
    archivos_validos = True
    archivos_validos &= validar_archivo(CSV_CRYPTO, "crypto_raw.csv")
    archivos_validos &= validar_archivo(CSV_FEATURES, "features_clustering.csv")
    archivos_validos &= validar_archivo(CSV_CLUSTERS, "cluster_labels.csv")
    
    if not archivos_validos:
        print("\n[ERROR] No se pueden continuar sin los archivos requeridos.")
        sys.exit(1)
    
    print(f"\n[OUTPUT] Las visualizaciones se guardaran en: {OUTPUT_DIR}\n")
    
    # Cargar datos
    print("[CARGA] Leyendo archivos CSV...")
    df_crypto = cargar_datos(CSV_CRYPTO)
    df_features = cargar_datos(CSV_FEATURES)
    df_clusters = cargar_datos(CSV_CLUSTERS)
    
    if df_crypto is None or df_features is None or df_clusters is None:
        print("\n[ERROR] No se pudieron cargar todos los datos.")
        sys.exit(1)
    
    # Generar visualizaciones EDA
    print("\n[EDA] Generando visualizaciones de analisis exploratorio...")
    plot_price_timeseries(df_crypto)
    plot_price_normalized(df_crypto)
    plot_returns_distribution(df_crypto)
    plot_correlation_heatmap(df_crypto)
    plot_volume_analysis(df_crypto)
    plot_rolling_volatility(df_crypto)
    plot_seasonality_month(df_crypto)
    plot_seasonality_dayofweek(df_crypto)
    plot_outliers_detection(df_crypto)
    
    # Generar visualizaciones de Clustering
    print("\n[CLUSTERING] Generando visualizaciones de clustering...")
    plot_k_selection_metrics(df_features)
    plot_pca_clustering(df_features, df_clusters)
    plot_features_heatmap(df_features, df_clusters)
    plot_cluster_statistics(df_features, df_clusters)
    
    print("\n" + "="*70)
    print(" COMPLETADO: Todas las visualizaciones fueron generadas.")
    print(f" Directorio de salida: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

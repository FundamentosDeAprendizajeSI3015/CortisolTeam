"""
Alejandro Garcés && Jean Londoño
[02] EDA — Exploratory Data Analysis
Generates statistics, figures, and exports findings to reports/.

Usage:
    python src/eda.py

Outputs:
    reports/figures/*.png   — 12 figures
    reports/eda_summary.md  — summary of findings (generated separately)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parent.parent
# Path to the raw crypto data CSV file
DATA_PATH = ROOT / 'data' / 'crypto_raw.csv'
# Directory for saving figures
FIGURES_DIR = ROOT / 'reports' / 'figures'
# Create figures directory if it doesn't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Main cryptocurrencies to analyze
MONEDAS_PRINCIPALES = ['BTC', 'ETH', 'BNB']
# Colors for each cryptocurrency
COLORES = {'BTC': '#F7931A', 'ETH': '#627EEA', 'BNB': '#F3BA2F'}
# Spanish month abbreviations (used in plots)
MESES_ES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
# Spanish day abbreviations (used in plots)
DIAS_ES = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
# Plotly template for consistent styling
PLOTLY_TEMPLATE = 'plotly_white'

# Set seaborn theme for plots
sns.set_theme(style='whitegrid', palette='muted')

# ── Data Loading ──────────────────────────────────────────────────────────────────

def cargar_datos(path: Path) -> pd.DataFrame:
    """Loads and prepares the raw dataset for analysis.

    Args:
        path: Path to the crypto_raw.csv file.

    Returns:
        Sorted DataFrame with calculated Return column.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {path}')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    df['Return'] = df.groupby('Symbol')['Close'].pct_change()
    return df

# ── Descriptive Statistics ──────────────────────────────────────────────

def estadisticas_por_moneda(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean, std, skew, and kurtosis per symbol and variable.

    Args:
        df: DataFrame with columns Symbol, Close, Volume, Marketcap, Return.

    Returns:
        DataFrame with index (Symbol, Variable) and statistics columns.
    """
    required = {'Symbol', 'Close', 'Volume', 'Marketcap', 'Return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    rows = []
    for sym, g in df.groupby('Symbol'):
        for col in ['Close', 'Volume', 'Marketcap', 'Return']:
            s = g[col].dropna()
            rows.append({
                'Symbol': sym, 'Variable': col,
                'mean': s.mean(), 'std': s.std(),
                'skew': s.skew(), 'kurtosis': s.kurtosis(),
                'min': s.min(), 'max': s.max(),
            })
    return pd.DataFrame(rows).set_index(['Symbol', 'Variable'])

# ── Time Series ───────────────────────────────────────────────────────

def plot_series_tiempo(df: pd.DataFrame) -> None:
    """Generates and saves time series plots of price, normalized price, and volume.

    Args:
        df: DataFrame with columns Symbol, Date, Close, Volume.
    """
    required = {'Symbol', 'Date', 'Close', 'Volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    df_top = df[df['Symbol'].isin(MONEDAS_PRINCIPALES)].copy()

    fig = go.Figure()
    for sym in MONEDAS_PRINCIPALES:
        d = df_top[df_top['Symbol'] == sym]
        fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name=sym,
                                 line=dict(color=COLORES[sym], width=1.5),
                                 hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title='Closing Price — BTC, ETH, BNB',
                      xaxis_title='Date', yaxis_title='USD',
                      template=PLOTLY_TEMPLATE, hovermode='x unified', height=450)
    fig.write_image(str(FIGURES_DIR / '01_price_timeseries.png'), width=1400, height=500, scale=2)
    print('[ok] 01_price_timeseries.png')

    fig2 = go.Figure()
    for sym in MONEDAS_PRINCIPALES:
        d = df_top[df_top['Symbol'] == sym].copy()
        d['Close_norm'] = d['Close'] / d['Close'].iloc[0] * 100
        fig2.add_trace(go.Scatter(x=d['Date'], y=d['Close_norm'], name=sym,
                                  line=dict(color=COLORES[sym], width=1.5)))
    fig2.update_layout(title='Normalized Price (base=100)',
                       template=PLOTLY_TEMPLATE, height=450)
    fig2.write_image(str(FIGURES_DIR / '02_price_normalized.png'), width=1400, height=500, scale=2)
    print('[ok] 02_price_normalized.png')

    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                         subplot_titles=[f'Trading Volume — {s}' for s in MONEDAS_PRINCIPALES])
    for i, sym in enumerate(MONEDAS_PRINCIPALES, 1):
        d = df_top[df_top['Symbol'] == sym]
        fig3.add_trace(go.Bar(x=d['Date'], y=d['Volume'], name=sym,
                              marker_color=COLORES[sym], opacity=0.7), row=i, col=1)
    # Use logarithmic scale on y-axis to handle wide range of volumes (from near-zero to billions)
    fig3.update_yaxes(type='log', row=1, col=1)
    fig3.update_yaxes(type='log', row=2, col=1)
    fig3.update_yaxes(type='log', row=3, col=1)
    fig3.update_layout(title='Trading Volume (logarithmic scale)', template=PLOTLY_TEMPLATE,
                       showlegend=False, height=700)
    fig3.write_image(str(FIGURES_DIR / '03_volume_timeseries.png'), width=1400, height=700, scale=2)
    print('[ok] 03_volume_timeseries.png')

# ── Correlation ────────────────────────────────────────────────────────────

def plot_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """Generates correlation heatmap of daily returns between cryptocurrencies.

    Args:
        df: DataFrame with columns Date, Symbol, Return.

    Returns:
        Pearson correlation matrix.
    """
    required = {'Date', 'Symbol', 'Return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    pivot = df.pivot_table(index='Date', columns='Symbol', values='Return')
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax,
                annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
    ax.set_title('Daily Returns Correlation between Cryptocurrencies', fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 04_correlation_heatmap.png')
    return corr

# ── Rolling Volatility ────────────────────────────────────────────────────

def calcular_volatilidad_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Adds vol_30d and vol_90d columns to the DataFrame.

    Args:
        df: DataFrame with columns Symbol, Return.

    Returns:
        DataFrame with added vol_30d and vol_90d columns.
    """
    required = {'Symbol', 'Return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    df = df.copy()
    df['vol_30d'] = df.groupby('Symbol')['Return'].transform(
        lambda x: x.rolling(30, min_periods=15).std())
    df['vol_90d'] = df.groupby('Symbol')['Return'].transform(
        lambda x: x.rolling(90, min_periods=45).std())
    return df


def plot_volatilidad(df: pd.DataFrame) -> None:
    """Generates rolling volatility plot for main cryptocurrencies.

    Args:
        df: DataFrame with columns Symbol, Date, vol_30d, vol_90d.
    """
    required = {'Symbol', 'Date', 'vol_30d', 'vol_90d'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f'Volatility — {s}' for s in MONEDAS_PRINCIPALES])
    for i, sym in enumerate(MONEDAS_PRINCIPALES, 1):
        d = df[df['Symbol'] == sym]
        fig.add_trace(go.Scatter(x=d['Date'], y=d['vol_30d'], name='Std 30d',
                                 line=dict(color='steelblue', width=1.2),
                                 showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=d['Date'], y=d['vol_90d'], name='Std 90d',
                                 line=dict(color='tomato', width=1.2, dash='dot'),
                                 showlegend=(i == 1)), row=i, col=1)
    fig.update_layout(title='Rolling Volatility (Std of daily returns)',
                      template=PLOTLY_TEMPLATE, height=750)
    fig.write_image(str(FIGURES_DIR / '05_volatilidad_rolling.png'), width=1400, height=750, scale=2)
    print('[ok] 05_volatilidad_rolling.png')


# ── Outliers ───────────────────────────────────────────────────────────────

def detectar_outliers(serie: pd.Series, metodo: str = 'zscore',
                      umbral: float = 3.0) -> pd.Series:
    """Detects outliers in a series using Z-Score or IQR.

    Args:
        serie: Numeric series of returns.
        metodo: 'zscore' or 'iqr'.
        umbral: Z-Score threshold (default 3.0).

    Returns:
        Boolean mask with True where outlier is detected.
    """
    s = serie.dropna()
    mask = pd.Series(False, index=serie.index)
    if metodo == 'zscore':
        mask.loc[s.index] = np.abs(stats.zscore(s)) > umbral
    elif metodo == 'iqr':
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask.loc[s.index] = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    else:
        raise ValueError(f'Unknown method: {metodo}. Use "zscore" or "iqr".')
    return mask

def plot_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Adds outlier columns and generates figures for BTC.

    Args:
        df: DataFrame with columns Symbol, Date, Close, Return.

    Returns:
        DataFrame with added outlier_z and outlier_iqr columns.
    """
    required = {'Symbol', 'Date', 'Close', 'Return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    df = df.copy()
    df['outlier_z'] = df.groupby('Symbol')['Return'].transform(
        lambda x: detectar_outliers(x, 'zscore'))
    df['outlier_iqr'] = df.groupby('Symbol')['Return'].transform(
        lambda x: detectar_outliers(x, 'iqr'))

    btc = df[df['Symbol'] == 'BTC'].copy()
    btc_z = btc[btc['outlier_z']]
    btc_iqr = btc[btc['outlier_iqr'] & ~btc['outlier_z']]

    for col_y, fname, title in [
        ('Close', '06_outliers_btc.png', 'BTC — Closing Price with Outliers'),
        ('Return', '07_outliers_retornos_btc.png', 'BTC — Daily Returns with Outliers'),
    ]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=btc['Date'], y=btc[col_y], mode='lines',
                                 name='BTC', line=dict(color='#F7931A', width=1.5), opacity=0.8))
        fig.add_trace(go.Scatter(x=btc_z['Date'], y=btc_z[col_y], mode='markers',
                                 name='Z-Score Outlier',
                                 marker=dict(color='red', size=8, symbol='x')))
        fig.add_trace(go.Scatter(x=btc_iqr['Date'], y=btc_iqr[col_y], mode='markers',
                                 name='IQR Outlier (only)',
                                 marker=dict(color='orange', size=7, symbol='triangle-up')))
        fig.update_layout(title=title, template=PLOTLY_TEMPLATE, height=500)
        fig.write_image(str(FIGURES_DIR / fname), width=1400, height=550, scale=2)
        print(f'[ok] {fname}')

    return df

# ── Histograms + QQ-plot ──────────────────────────────────────────────────

def plot_hist_qq(df: pd.DataFrame, symbols: list) -> None:
    """Generates histogram of returns and QQ-plot for the specified symbols.

    Args:
        df: DataFrame with columns Symbol and Return.
        symbols: List of symbols to plot.
    """
    required = {'Symbol', 'Return'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    n = len(symbols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))
    if n == 1:
        axes = [axes]

    for i, sym in enumerate(symbols):
        retornos = df[df['Symbol'] == sym]['Return'].dropna()

        ax_hist = axes[i][0]
        sns.histplot(retornos, bins=60, kde=True, ax=ax_hist,
                     color='steelblue', stat='density')
        x_norm = np.linspace(retornos.min(), retornos.max(), 200)
        ax_hist.plot(x_norm, stats.norm.pdf(x_norm, retornos.mean(), retornos.std()),
                     'r--', linewidth=1.5, label='Theoretical Normal')
        ax_hist.set_title(f'{sym} — Histogram of Daily Returns')
        ax_hist.set_xlabel('Daily Return')
        ax_hist.set_ylabel('Density')
        sk, ku = retornos.skew(), retornos.kurtosis()
        ax_hist.text(0.97, 0.95, f'Skew={sk:.2f}\nKurt={ku:.2f}',
                     transform=ax_hist.transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round', alpha=0.3))
        ax_hist.legend(fontsize=8)

        ax_qq = axes[i][1]
        (osm, osr), (slope, intercept, r) = stats.probplot(retornos, dist='norm')
        ax_qq.scatter(osm, osr, s=8, alpha=0.4, color='steelblue', label='Data')
        x_line = np.array([min(osm), max(osm)])
        ax_qq.plot(x_line, slope * x_line + intercept, 'r--',
                   linewidth=1.5, label='Normal reference')
        ax_qq.set_title(f'{sym} — QQ-plot (Normal)')
        ax_qq.set_xlabel('Theoretical Quantiles')
        ax_qq.set_ylabel('Observed Quantiles')
        ax_qq.text(0.03, 0.95, f'R²={r**2:.3f}',
                   transform=ax_qq.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', alpha=0.3))
        ax_qq.legend(fontsize=8)

    plt.suptitle('Distribution of Daily Returns', y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '08_hist_qq.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 08_hist_qq.png')

    fig_box, ax = plt.subplots(figsize=(16, 6))
    orden = df.groupby('Symbol')['Return'].median().sort_values().index.tolist()
    sns.boxplot(data=df.dropna(subset=['Return']), x='Symbol', y='Return',
                order=orden, ax=ax, showfliers=False, palette='muted')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Distribution of Daily Returns by Currency', fontsize=13)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig_box.savefig(FIGURES_DIR / '09_boxplot_retornos.png', dpi=150, bbox_inches='tight')
    plt.close(fig_box)
    print('[ok] 09_boxplot_retornos.png')

# ── Seasonality ────────────────────────────────────────────────────────────

def plot_seasonality(df: pd.DataFrame, symbols: list) -> None:
    """Generates seasonality figures by month and day of week.

    Args:
        df: DataFrame with columns Symbol, Return, mes, dia_semana.
        symbols: List of symbols to plot.
    """
    required = {'Symbol', 'Return', 'mes', 'dia_semana'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    n = len(symbols)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, sym in zip(axes, symbols):
        d = df[df['Symbol'] == sym].dropna(subset=['Return'])
        mes_avg = d.groupby('mes')['Return'].mean().reindex(range(1, 13))
        colores = ['tomato' if v < 0 else 'seagreen' for v in mes_avg.fillna(0)]
        ax.bar(range(1, 13), mes_avg.values, color=colores)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MESES_ES, rotation=45, fontsize=9)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'{sym} — Average Return by Month', fontsize=11)
        ax.set_ylabel('Average Return')
    plt.suptitle('Monthly Seasonality', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '10_seasonality_mes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 10_seasonality_mes.png')

    fig2, axes2 = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes2 = [axes2]
    for ax, sym in zip(axes2, symbols):
        d = df[df['Symbol'] == sym].dropna(subset=['Return'])
        dia_avg = d.groupby('dia_semana')['Return'].mean().reindex(range(7))
        colores = ['tomato' if v < 0 else 'steelblue' for v in dia_avg.fillna(0)]
        ax.bar(range(7), dia_avg.values, color=colores)
        ax.set_xticks(range(7))
        ax.set_xticklabels(DIAS_ES, rotation=45, fontsize=9)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'{sym} — Average Return by Day', fontsize=11)
        ax.set_ylabel('Average Return')
    plt.suptitle('Day of Week Seasonality', fontsize=13, y=1.02)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / '11_seasonality_dia.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('[ok] 11_seasonality_dia.png')

    pivot_mes = (df.dropna(subset=['Return'])
                 .groupby(['Symbol', 'mes'])['Return'].mean()
                 .unstack('mes'))
    pivot_mes.columns = MESES_ES
    fig3, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot_mes, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                linewidths=0.3, ax=ax, annot_kws={'size': 8},
                cbar_kws={'label': 'Average Monthly Return'})
    ax.set_title('Average Return by Month and Currency', fontsize=13)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / '12_heatmap_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print('[ok] 12_heatmap_seasonality.png')

def main():
    """Runs the complete EDA pipeline."""
    print('[EDA] Loading data ...')
    df = cargar_datos(DATA_PATH)
    print(f'  Shape: {df.shape} | Currencies: {df["Symbol"].nunique()} | '
          f'Range: {df["Date"].min().date()} -> {df["Date"].max().date()}')

    print('\n[EDA] Descriptive statistics ...')
    stats_df = estadisticas_por_moneda(df)
    print(stats_df.xs('Return', level='Variable')
          .sort_values('std', ascending=False)
          .round(4)
          .to_string())

    print('\n[EDA] Time series ...')
    plot_series_tiempo(df)

    print('\n[EDA] Correlation ...')
    plot_correlacion(df)

    print('\n[EDA] Rolling volatility ...')
    df = calcular_volatilidad_rolling(df)
    plot_volatilidad(df)

    print('\n[EDA] Outliers ...')
    df = plot_outliers(df)

    print('\n[EDA] Histograms + QQ-plot ...')
    plot_hist_qq(df, MONEDAS_PRINCIPALES)

    print('\n[EDA] Seasonality ...')
    df['mes'] = df['Date'].dt.month
    df['dia_semana'] = df['Date'].dt.dayofweek
    plot_seasonality(df, MONEDAS_PRINCIPALES)

    print(f'\n[done] EDA completed. Figures in {FIGURES_DIR}')


if __name__ == '__main__':
    main()
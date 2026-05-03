"""
[03] Feature Engineering — Crypto ML Project
Agrega indicadores técnicos y crea el target de clasificación binaria.

Uso:
<<<<<<< HEAD
    python supervised/feature_engineering.py
=======
    python src/feature_engineering.py
>>>>>>> origin/desarrollo

Salida:
    data/processed/features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / 'data' / 'crypto_raw.csv'
OUT_PATH   = ROOT / 'data' / 'processed' / 'features.parquet'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Monedas excluidas: stablecoins (sin señal de precio) y WBTC (colineal con BTC)
SYMBOLS_EXCLUIR = {'USDT', 'USDC', 'WBTC'}


def cargar_datos(path: Path) -> pd.DataFrame:
    """Carga el dataset crudo y lo prepara para feature engineering.

    Args:
        path: Ruta a crypto_raw.csv.

    Returns:
        DataFrame ordenado por Symbol y Date con columna Return calculada.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f'No se encontró el archivo: {path}')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    simbolos_validos = ~df['Symbol'].isin(SYMBOLS_EXCLUIR)
    df = df[simbolos_validos].copy()

    df['Return'] = df.groupby('Symbol')['Close'].pct_change()
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores técnicos por símbolo al DataFrame.

    Indicadores añadidos:
        - SMA 7, 14, 30 días
        - EMA 14 días
        - RSI 14 días
        - Bollinger Bands 20 días (banda alta, baja y ancho)
        - MACD (línea MACD y señal)

    Args:
        df: DataFrame con columnas Symbol, Date, Close, High, Low, Volume.

    Returns:
        DataFrame con columnas de indicadores técnicos añadidas.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    required = {'Symbol', 'Date', 'Close', 'High', 'Low', 'Volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Columnas requeridas faltantes: {missing}')

    grupos = []
    for sym, g in df.groupby('Symbol', sort=False):
        g = g.copy().sort_values('Date').reset_index(drop=True)
        close = g['Close']
        high  = g['High']
        low   = g['Low']

        # SMA
        g['sma_7']  = SMAIndicator(close, window=7,  fillna=False).sma_indicator()
        g['sma_14'] = SMAIndicator(close, window=14, fillna=False).sma_indicator()
        g['sma_30'] = SMAIndicator(close, window=30, fillna=False).sma_indicator()

        # EMA
        g['ema_14'] = EMAIndicator(close, window=14, fillna=False).ema_indicator()

        # RSI
        g['rsi_14'] = RSIIndicator(close, window=14, fillna=False).rsi()

        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2, fillna=False)
        g['bb_high']  = bb.bollinger_hband()
        g['bb_low']   = bb.bollinger_lband()
        g['bb_width'] = bb.bollinger_wband()

        # MACD
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        g['macd']        = macd.macd()
        g['macd_signal'] = macd.macd_signal()

        grupos.append(g)

    return pd.concat(grupos, ignore_index=True).sort_values(['Symbol', 'Date'])


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la columna target: 1 si Close(t+1) > Close(t), 0 si no.

    Args:
        df: DataFrame con columnas Symbol y Close.

    Returns:
        DataFrame con columna target añadida (NaN en el último registro de cada símbolo).

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    required = {'Symbol', 'Close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Columnas requeridas faltantes: {missing}')

    df = df.copy()
    close_next = df.groupby('Symbol')['Close'].shift(-1)
    df['target'] = (close_next > df['Close']).astype(float)
    df.loc[df.groupby('Symbol')['Close'].tail(1).index, 'target'] = float('nan')
    return df


def main():
    """Ejecuta el pipeline de feature engineering y guarda el resultado."""
    print('[feature_engineering] Cargando datos ...')
    df = cargar_datos(DATA_PATH)
    print(f'  Shape inicial: {df.shape} | Monedas: {sorted(df["Symbol"].unique())}')

    print('\n[feature_engineering] Calculando indicadores técnicos ...')
    df = add_technical_indicators(df)

    print('\n[feature_engineering] Creando label binario ...')
    df = create_binary_label(df)

    # Eliminar filas sin suficientes datos históricos (NaN en indicadores)
    n_antes = len(df)
    df = df.dropna(subset=['sma_30', 'rsi_14', 'macd', 'target'])
    n_despues = len(df)
    print(f'  Filas eliminadas por NaN en indicadores/target: {n_antes - n_despues}')
    print(f'  Shape final: {df.shape}')
    print(f'  Balance de clases: {df["target"].value_counts().to_dict()}')

    print(f'\n[feature_engineering] Guardando en {OUT_PATH} ...')
    df.to_parquet(OUT_PATH, index=False)
    print(f'[done] {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()

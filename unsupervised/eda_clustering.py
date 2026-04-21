"""
[03] EDA para Clustering — Crypto ML Project
Prepara el feature matrix por moneda listo para clustering.

Uso:
    python unsupervised/eda_clustering.py

Salida:
    data/features_clustering.csv   — features normalizados por moneda
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CSV_IN  = DATA_DIR / "crypto_raw.csv"
CSV_OUT = DATA_DIR / "features_clustering.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_IN, parse_dates=["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    print(f"[load] {df.shape[0]} filas, {df['Symbol'].nunique()} monedas")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye un feature por moneda para clustering."""
    records = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()
        close = group["Close"]

        daily_return = close.pct_change().dropna()

        volatility_30d  = daily_return.rolling(30).std().mean()
        mean_return     = daily_return.mean()
        sharpe_ratio    = (mean_return / daily_return.std()) * np.sqrt(365) if daily_return.std() > 0 else 0

        cumulative      = (1 + daily_return).cumprod()
        rolling_max     = cumulative.cummax()
        drawdown        = (cumulative - rolling_max) / rolling_max
        max_drawdown    = drawdown.min()

        avg_volume      = group["Volume"].mean()

        records.append({
            "Symbol":       symbol,
            "mean_return":  mean_return,
            "volatility":   volatility_30d,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_volume":   avg_volume,
        })

    features = pd.DataFrame(records).set_index("Symbol")
    print(f"[features] Shape: {features.shape}")
    print(features.describe().round(4))
    return features


def normalize(features: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns,
    )
    return scaled


def main():
    df       = load_data()
    features = build_features(df)
    scaled   = normalize(features)
    scaled.to_csv(CSV_OUT)
    print(f"\n[save] {CSV_OUT}")
    print("[done] EDA para clustering completado.")


if __name__ == "__main__":
    main()

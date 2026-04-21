"""
[01] Load Data — Crypto ML Project
Descarga, valida e ingesta el dataset Cryptocurrency Historical Prices (Kaggle).

Uso:
    python load_data/load_data.py

Salida:
    data/crypto_raw.csv
"""

import hashlib
import kagglehub
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_OUT = DATA_DIR / "crypto_raw.csv"


def download_dataset() -> Path:
    print("[download] Descargando dataset ...")
    path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")
    print(f"[ok] Dataset en: {path}")
    return Path(path)


def load_csvs(dataset_path: Path) -> pd.DataFrame:
    csv_files = sorted(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en {dataset_path}")

    print(f"[load] {len(csv_files)} archivos CSV encontrados")
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[ok] DataFrame combinado: {combined.shape}")
    return combined


def validate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[validate] Iniciando validación ...")

    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    nat_count = df["Date"].isna().sum()
    if nat_count:
        print(f"  [warn] {nat_count} filas con Date inválida — se eliminan")
        df = df.dropna(subset=["Date"])

    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Marketcap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("-", "", regex=False)
                .replace("", float("nan"))
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dupes = df.duplicated(subset=["Symbol", "Date"]).sum()
    if dupes:
        print(f"  [warn] {dupes} duplicados eliminados")
        df = df.drop_duplicates(subset=["Symbol", "Date"])

    print(f"  Shape final : {df.shape}")
    print(f"  Monedas     : {sorted(df['Symbol'].unique())}")
    print(f"  Rango fechas: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print("[ok] Validación completada\n")
    return df


def save(df: pd.DataFrame):
    df.to_csv(CSV_OUT, index=False)
    md5 = hashlib.md5(CSV_OUT.read_bytes()).hexdigest()
    print(f"[save] {CSV_OUT}  ({CSV_OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"  MD5: {md5}")


def main():
    dataset_path = download_dataset()
    df = load_csvs(dataset_path)
    df = validate(df)
    save(df)
    print("\n[done] Load Data completado.")


if __name__ == "__main__":
    main()

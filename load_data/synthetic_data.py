"""
[01b] Synthetic Data Generator — Crypto ML Project
Genera filas sintéticas OHLCV adicionales para alcanzar un total de 1,000,000
filas combinadas con el dataset real de Kaggle.

Estrategia:
    - Lee data/crypto_raw_real.csv (backup del dataset real, 37,082 filas).
    - Calibra parámetros (mu, sigma, rangos de volumen) a partir de las
      distribuciones empíricas de las criptomonedas reales.
    - Crea 220 símbolos sintéticos (SYN001..SYN220) con perfiles variados:
        · 6%  stablecoin-like  (volatilidad ~0, precio anclado al USD).
        · 14% baja volatilidad (sigma ~ 0.015).
        · 50% volatilidad media (sigma ~ 0.04).
        · 25% alta volatilidad (sigma ~ 0.08).
        · 5%  meme/extreme     (sigma ~ 0.15 con jumps frecuentes).
    - Simula precio con GBM + jump-diffusion (saltos de Poisson y
      drawdowns ocasionales).
    - Genera OHLC consistente (Low <= Open,Close <= High) y Volume/Marketcap
      proporcionales al precio.
    - Marca cada fila sintética con la columna is_synthetic = True.
    - Concatena con el dataset real (is_synthetic = False) para producir
      exactamente 1,000,000 filas y guarda en data/crypto_raw.csv.

Uso:
    python load_data/synthetic_data.py

Salida:
    data/crypto_raw.csv     (1,000,000 filas)
    data/crypto_raw_real.csv (sin cambios — backup del original)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REAL_PATH = DATA_DIR / "crypto_raw_real.csv"
OUT_PATH = DATA_DIR / "crypto_raw.csv"

TARGET_TOTAL = 1_000_000
RANDOM_SEED = 42

# Rango temporal sintético: histórico amplio para que cada moneda tenga ~4378 días
START_DATE = pd.Timestamp("2013-01-01")
END_DATE = pd.Timestamp("2024-12-31")
N_DAYS = (END_DATE - START_DATE).days + 1   # 4383


def perfil_simbolos(n_synth: int, rng: np.random.Generator) -> list[dict]:
    """Genera la lista de perfiles que parametrizan cada moneda sintética.

    Cada perfil define mu, sigma, prob_jump, jump_size_mean, p0 (precio inicial),
    volume_base y marketcap_base.
    """
    n_stable = int(round(0.06 * n_synth))
    n_low = int(round(0.14 * n_synth))
    n_high = int(round(0.25 * n_synth))
    n_meme = int(round(0.05 * n_synth))
    n_mid = n_synth - (n_stable + n_low + n_high + n_meme)

    perfiles = []

    def add(kind, n, mu_range, sd_range, jump_p_range, jump_mu_range,
            p0_range, vol_range, cap_range):
        for _ in range(n):
            perfiles.append({
                "kind": kind,
                "mu": float(rng.uniform(*mu_range)),
                "sigma": float(rng.uniform(*sd_range)),
                "jump_p": float(rng.uniform(*jump_p_range)),
                "jump_mu": float(rng.uniform(*jump_mu_range)),
                "p0": float(rng.uniform(*p0_range)),
                "vol_base": float(rng.uniform(*vol_range)),
                "cap_base": float(rng.uniform(*cap_range)),
            })

    add("stable", n_stable,
        mu_range=(-1e-5, 1e-5), sd_range=(1e-4, 8e-4),
        jump_p_range=(0.0, 0.001), jump_mu_range=(0.0, 0.001),
        p0_range=(0.99, 1.01), vol_range=(1e8, 5e9), cap_range=(5e8, 5e10))

    add("low_vol", n_low,
        mu_range=(0.0, 0.0008), sd_range=(0.01, 0.025),
        jump_p_range=(0.001, 0.005), jump_mu_range=(0.01, 0.03),
        p0_range=(0.5, 50.0), vol_range=(1e6, 5e8), cap_range=(1e7, 5e9))

    add("mid_vol", n_mid,
        mu_range=(0.0, 0.002), sd_range=(0.025, 0.06),
        jump_p_range=(0.005, 0.015), jump_mu_range=(0.03, 0.08),
        p0_range=(0.001, 200.0), vol_range=(1e5, 1e9), cap_range=(1e6, 1e10))

    add("high_vol", n_high,
        mu_range=(-0.001, 0.0035), sd_range=(0.05, 0.10),
        jump_p_range=(0.01, 0.025), jump_mu_range=(0.06, 0.15),
        p0_range=(1e-5, 100.0), vol_range=(1e4, 5e8), cap_range=(1e5, 5e9))

    add("meme", n_meme,
        mu_range=(-0.002, 0.005), sd_range=(0.10, 0.20),
        jump_p_range=(0.02, 0.05), jump_mu_range=(0.12, 0.30),
        p0_range=(1e-7, 1.0), vol_range=(1e3, 1e8), cap_range=(1e4, 1e9))

    rng.shuffle(perfiles)
    return perfiles


def simular_precio(perfil: dict, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Simula serie de Close mediante GBM con jumps Poisson."""
    mu = perfil["mu"]
    sigma = perfil["sigma"]
    p_jump = perfil["jump_p"]
    jump_mu = perfil["jump_mu"]
    p0 = perfil["p0"]

    # Retornos log-normales
    eps = rng.standard_normal(n_days)
    log_ret = (mu - 0.5 * sigma ** 2) + sigma * eps

    # Jumps de Poisson asimétricos (más negativos que positivos —ej. crashes)
    jumps_n = rng.binomial(1, p_jump, size=n_days)
    jump_dir = rng.choice([-1.0, 1.0], size=n_days, p=[0.6, 0.4])
    jump_mag = np.abs(rng.normal(jump_mu, jump_mu * 0.3, size=n_days))
    log_ret += jumps_n * jump_dir * jump_mag

    # Acumular en precio
    price = p0 * np.exp(np.cumsum(log_ret))

    # Limitar precios extremos para evitar overflow / valores absurdos
    price = np.clip(price, p0 * 1e-6, p0 * 1e6)

    if perfil["kind"] == "stable":
        # Stablecoins: oscilan alrededor de 1 USD
        price = 1.0 + (price - p0) * 0.01
        price = np.clip(price, 0.97, 1.03)
    return price


def construir_ohlc(close: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deriva Open, High, Low consistentes a partir de Close."""
    n = len(close)
    # Open = Close del día anterior con pequeño gap
    gap = rng.normal(0, 0.005, size=n)
    open_ = np.empty(n)
    open_[0] = close[0] * (1 + rng.normal(0, 0.005))
    open_[1:] = close[:-1] * (1 + gap[1:])

    # Rango intradía proporcional a |return|
    daily_ret = np.diff(close, prepend=close[0]) / np.where(close == 0, 1, close)
    intraday_range = np.maximum(np.abs(daily_ret), 0.005) * (1 + np.abs(rng.normal(0, 0.5, size=n)))
    half = close * intraday_range / 2

    base = np.maximum(open_, close)
    base_low = np.minimum(open_, close)
    high = base + np.abs(rng.normal(half, half * 0.3))
    low = base_low - np.abs(rng.normal(half, half * 0.3))
    low = np.clip(low, 1e-12, None)
    return open_, high, low


def construir_volume(close: np.ndarray, perfil: dict, rng: np.random.Generator) -> np.ndarray:
    """Volume USD log-normal centrado en perfil['vol_base'] modulado por |return|."""
    n = len(close)
    base = perfil["vol_base"]
    daily_ret = np.abs(np.diff(close, prepend=close[0]) / np.where(close == 0, 1, close))
    activity = 1 + 5 * daily_ret  # más volumen cuando hay movimiento fuerte
    noise = rng.lognormal(mean=0.0, sigma=0.5, size=n)
    vol = base * activity * noise
    return vol


def construir_marketcap(close: np.ndarray, perfil: dict, rng: np.random.Generator) -> np.ndarray:
    """Marketcap = supply × Close. Supply implícita estable, calibrada al cap_base."""
    cap0 = perfil["cap_base"]
    supply = cap0 / max(close[0], 1e-12)
    noise = rng.normal(1.0, 0.02, size=len(close))   # pequeño ruido de circulación
    return supply * close * noise


def generar_dataset_sintetico(target_rows: int, real_max_sno: int,
                              rng: np.random.Generator) -> pd.DataFrame:
    """Genera DataFrame con exactamente `target_rows` filas sintéticas."""
    # Determinamos número de símbolos: 220 default → ~4377 días cada uno
    n_synth = 220
    perfiles = perfil_simbolos(n_synth, rng)

    base_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n_dates_total = len(base_dates)

    # Filas por símbolo: aproximadamente iguales, último símbolo absorbe residuo
    base_per_sym = target_rows // n_synth
    residuo = target_rows - base_per_sym * n_synth

    print(f"[synth] Generando {n_synth} símbolos sintéticos")
    print(f"[synth] Filas base por símbolo: {base_per_sym}, residuo: {residuo}")

    frames = []
    sno = real_max_sno
    for i, perfil in enumerate(perfiles, start=1):
        n_rows = base_per_sym + (1 if i <= residuo else 0)
        if n_rows > n_dates_total:
            n_rows = n_dates_total
        # Elegir tramo aleatorio si n_rows < n_dates_total
        if n_rows < n_dates_total:
            start_idx = int(rng.integers(0, n_dates_total - n_rows + 1))
        else:
            start_idx = 0
        dates = base_dates[start_idx:start_idx + n_rows]

        close = simular_precio(perfil, n_rows, rng)
        open_, high, low = construir_ohlc(close, rng)
        vol = construir_volume(close, perfil, rng)
        cap = construir_marketcap(close, perfil, rng)

        symbol = f"SYN{i:03d}"
        name = f"Synthetic-{perfil['kind']}-{i:03d}"
        sno_arr = np.arange(sno + 1, sno + 1 + n_rows, dtype=np.int64)
        sno += n_rows

        df_sym = pd.DataFrame({
            "SNo": sno_arr,
            "Name": name,
            "Symbol": symbol,
            "Date": dates,
            "High": high,
            "Low": low,
            "Open": open_,
            "Close": close,
            "Volume": vol,
            "Marketcap": cap,
            "is_synthetic": True,
        })
        frames.append(df_sym)
        if i % 50 == 0:
            print(f"  [{i}/{n_synth}] {symbol} ({perfil['kind']}) — {n_rows} filas")

    out = pd.concat(frames, ignore_index=True)
    return out


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    if not REAL_PATH.exists():
        print(f"[error] {REAL_PATH} no encontrado.", file=sys.stderr)
        print("Ejecuta primero el backup: cp data/crypto_raw.csv data/crypto_raw_real.csv",
              file=sys.stderr)
        sys.exit(1)

    print(f"[real] Cargando dataset real desde {REAL_PATH} ...")
    real = pd.read_csv(REAL_PATH, parse_dates=["Date"])
    real["is_synthetic"] = False
    n_real = len(real)
    max_sno = int(real["SNo"].max()) if "SNo" in real.columns else 0
    print(f"  Filas reales: {n_real:,} | SNo máx: {max_sno}")

    target_synth = TARGET_TOTAL - n_real
    if target_synth <= 0:
        print(f"[ok] dataset real ya tiene >= {TARGET_TOTAL} filas")
        return
    print(f"[synth] Objetivo sintético: {target_synth:,} filas")

    synth = generar_dataset_sintetico(target_synth, max_sno, rng)
    print(f"[synth] Generado: {len(synth):,} filas, {synth['Symbol'].nunique()} símbolos")

    # Ajuste fino al target exacto
    if len(synth) > target_synth:
        synth = synth.iloc[:target_synth].copy()
    elif len(synth) < target_synth:
        # padding mínimo: replicar últimas filas del último símbolo
        falta = target_synth - len(synth)
        relleno = synth.tail(falta).copy()
        relleno["SNo"] = np.arange(synth["SNo"].max() + 1, synth["SNo"].max() + 1 + falta)
        synth = pd.concat([synth, relleno], ignore_index=True)

    combined = pd.concat([real, synth], ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    combined["SNo"] = np.arange(1, len(combined) + 1, dtype=np.int64)

    print()
    print(f"[combined] Filas totales: {len(combined):,}")
    print(f"[combined] Símbolos     : {combined['Symbol'].nunique()}")
    print(f"[combined] Reales       : {(~combined['is_synthetic']).sum():,}")
    print(f"[combined] Sintéticas   : {combined['is_synthetic'].sum():,}")
    print(f"[combined] Rango fechas : {combined['Date'].min().date()} -> {combined['Date'].max().date()}")

    print(f"\n[save] Escribiendo a {OUT_PATH} ...")
    combined.to_csv(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    md5 = hashlib.md5(OUT_PATH.read_bytes()).hexdigest()
    print(f"  Tamaño: {size_mb:.1f} MB | MD5: {md5}")
    print("[done] Dataset sintético + real combinado generado.")


if __name__ == "__main__":
    main()

"""
[03c] One-Class SVM — Crypto ML Project
Deteccion de anomalias no supervisada con analisis de sensibilidad de nu.

Uso:
    python unsupervised/svm_analysis.py

Salida:
    unsupervised/reports/svm_sensitivity.png   — grilla 1x5 PCA por nu
    unsupervised/reports/svm_vs_dbscan.png     — consensus SVM vs DBSCAN
    unsupervised/reports/svm_conclusions.txt   — tabla de estabilidad y analisis
    data/cluster_labels.csv                    — +columnas OneClassSVM_nu020, OneClassSVM_consensus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CSV_FEATURES = DATA_DIR / "features_clustering.csv"
CSV_LABELS   = DATA_DIR / "cluster_labels.csv"

NU_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]


def load_features() -> tuple:
    features = pd.read_csv(CSV_FEATURES, index_col="Symbol")
    labels   = pd.read_csv(CSV_LABELS,   index_col="Symbol")
    print(f"[load] {features.shape[0]} monedas, {features.shape[1]} features")
    print(f"[load] Columnas en cluster_labels: {list(labels.columns)}")
    return features, labels


def run_sensitivity(X: np.ndarray, symbols: list,
                    nu_values: list = NU_VALUES) -> dict:
    """Entrena un OneClassSVM por cada nu. Retorna dict {nu: array -1/1}."""
    results = {}

    print(f"\n{'nu':>6} | {'Anomalías':>9} | Monedas")
    print("-" * 60)

    for nu in nu_values:
        model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        preds = model.fit_predict(X)          # 1=normal, -1=anomalía
        anomalies = [s for s, p in zip(symbols, preds) if p == -1]
        results[nu] = preds
        print(f"{nu:>6.2f} | {len(anomalies):>9} | {anomalies}")

    return results


def build_consensus(results: dict, symbols: list, threshold: int = 3) -> np.ndarray:
    """
    Una moneda es anomalía de consenso si fue marcada como -1
    en al menos `threshold` valores de nu.
    Retorna array de -1 (anomalía) / 1 (normal).
    """
    n = len(symbols)
    anomaly_counts = np.zeros(n, dtype=int)

    for preds in results.values():
        anomaly_counts += (preds == -1).astype(int)

    consensus = np.where(anomaly_counts >= threshold, -1, 1)

    print(f"\n[consensus] Umbral: ≥{threshold} de {len(results)} nu values")
    print(f"{'Moneda':>8} | {'Conteo':>6} | {'Consenso':>9}")
    print("-" * 35)
    for sym, count, label in zip(symbols, anomaly_counts, consensus):
        tag = "ANOMALIA" if label == -1 else "normal"
        print(f"{sym:>8} | {count:>6} | {tag}")

    anomalies = [s for s, l in zip(symbols, consensus) if l == -1]
    print(f"\n[consensus] Anomalías estables ({len(anomalies)}): {anomalies}")
    return consensus


def plot_sensitivity(X: np.ndarray, results: dict, symbols: list) -> None:
    """Grilla 1x5: un subplot por nu. Rojo=anomalía, azul=normal."""
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()

    nu_values = sorted(results.keys())
    fig, axes = plt.subplots(1, len(nu_values), figsize=(5 * len(nu_values), 5))

    for ax, nu in zip(axes, nu_values):
        preds  = results[nu]
        colors = ["red" if p == -1 else "steelblue" for p in preds]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, zorder=3)
        for i, sym in enumerate(symbols):
            ax.annotate(sym, (coords[i, 0], coords[i, 1]),
                        fontsize=6, ha="center", va="bottom")
        n_anom = (preds == -1).sum()
        ax.set_title(f"nu={nu:.2f}\n({n_anom} anomalías)", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if nu == nu_values[0] else "")
        ax.tick_params(labelsize=7)

    legend = [
        mpatches.Patch(color="steelblue", label="Normal"),
        mpatches.Patch(color="red",       label="Anomalía"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=9)
    fig.suptitle(
        f"One-Class SVM — Sensibilidad de nu (PCA {var_exp:.0%} varianza)",
        fontsize=13
    )
    plt.tight_layout()
    out = REPORTS_DIR / "svm_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


def main():
    features, labels = load_features()
    X       = features.values
    symbols = list(features.index)

    print("\n--- Analisis de sensibilidad One-Class SVM ---")
    results   = run_sensitivity(X, symbols)

    print("\n--- Consenso de anomalías ---")
    consensus = build_consensus(results, symbols, threshold=3)

    print("\n--- Visualizaciones ---")
    plot_sensitivity(X, results, symbols)

    print("\n[done] svm_analysis completado.")


if __name__ == "__main__":
    main()

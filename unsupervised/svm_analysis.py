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


def main():
    features, labels = load_features()
    X       = features.values
    symbols = list(features.index)

    print("\n--- Analisis de sensibilidad One-Class SVM ---")
    results = run_sensitivity(X, symbols)

    print("\n[done] svm_analysis completado.")


if __name__ == "__main__":
    main()

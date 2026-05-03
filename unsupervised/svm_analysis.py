"""
[03c] One-Class SVM — Crypto ML Project
Deteccion de anomalias no supervisada con analisis de sensibilidad de nu.

Uso:
    python unsupervised/svm_analysis.py

Salida:
    unsupervised/reports/svm_sensitivity.png   — grilla 1x5 PCA por nu
    unsupervised/reports/svm_vs_dbscan.png     — consensus SVM vs DBSCAN
    unsupervised/reports/svm_report.txt        — tabla de estabilidad y analisis
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


def plot_vs_dbscan(X: np.ndarray, consensus: np.ndarray,
                   dbscan_labels: np.ndarray, symbols: list) -> None:
    """Dos subplots: One-Class SVM consensus vs DBSCAN. Misma paleta."""
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()

    # DBSCAN: -1=anomalía, >=0=cluster → convertir a -1/1 para misma paleta
    dbscan_binary = np.where(dbscan_labels == -1, -1, 1)
    n_agree = (consensus == dbscan_binary).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (preds, title) in zip(axes, [
        (consensus,     "One-Class SVM (consensus ≥3 nu)"),
        (dbscan_binary, "DBSCAN (referencia)"),
    ]):
        colors = ["red" if p == -1 else "steelblue" for p in preds]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=120, zorder=3,
                   edgecolors="k", linewidths=0.4)
        for i, sym in enumerate(symbols):
            ax.annotate(sym, (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha="center", va="bottom")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 (PCA {var_exp:.0%} varianza)")
        ax.set_ylabel("PC2")

    legend = [
        mpatches.Patch(color="steelblue", label="Normal"),
        mpatches.Patch(color="red",       label="Anomalía"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=9)
    fig.suptitle(
        f"Anomalías: One-Class SVM vs DBSCAN — Coincidencia: {n_agree}/{len(symbols)} monedas",
        fontsize=13
    )
    plt.tight_layout()
    out = REPORTS_DIR / "svm_vs_dbscan.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


def save_conclusions(results: dict, consensus: np.ndarray,
                     dbscan_labels: np.ndarray, symbols: list) -> None:
    """Genera svm_conclusions.txt con tabla de estabilidad y análisis comparativo."""
    nu_values = sorted(results.keys())
    dbscan_binary = np.where(dbscan_labels == -1, -1, 1)

    lines = []
    lines.append("=" * 60)
    lines.append("ANALISIS ONE-CLASS SVM — Crypto ML Project")
    lines.append("=" * 60)
    lines.append(f"\nDataset: {len(symbols)} monedas")
    lines.append(f"Nu values analizados: {nu_values}")
    lines.append(f"Umbral de consenso: ≥3 de {len(nu_values)} nu values\n")

    lines.append("-" * 60)
    lines.append("1. TABLA DE ESTABILIDAD POR MONEDA")
    lines.append("-" * 60)
    header = f"{'Moneda':>8} | " + " | ".join(f"nu={nu:.2f}" for nu in nu_values) + " | CONSENSO"
    lines.append(header)
    lines.append("-" * len(header))

    for i, sym in enumerate(symbols):
        row_vals = []
        count = 0
        for nu in nu_values:
            val = results[nu][i]
            row_vals.append("  ANOM" if val == -1 else "normal")
            if val == -1:
                count += 1
        consensus_tag = "ANOMALIA" if consensus[i] == -1 else "normal"
        lines.append(f"{sym:>8} | " + " | ".join(row_vals) + f" | {consensus_tag} ({count}/{len(nu_values)})")

    anom_svm  = [s for s, l in zip(symbols, consensus)      if l == -1]
    anom_dbs  = [s for s, l in zip(symbols, dbscan_binary)  if l == -1]
    coinciden = [s for s in anom_svm if s in anom_dbs]
    solo_svm  = [s for s in anom_svm if s not in anom_dbs]
    solo_dbs  = [s for s in anom_dbs if s not in anom_svm]

    lines.append(f"\n{'-' * 60}")
    lines.append("2. COMPARACION CON DBSCAN")
    lines.append("-" * 60)
    lines.append(f"  Anomalías One-Class SVM ({len(anom_svm)}): {anom_svm}")
    lines.append(f"  Anomalías DBSCAN         ({len(anom_dbs)}): {anom_dbs}")
    lines.append(f"  Coincidencias            ({len(coinciden)}): {coinciden}")
    lines.append(f"  Solo en SVM              ({len(solo_svm)}): {solo_svm}")
    lines.append(f"  Solo en DBSCAN           ({len(solo_dbs)}): {solo_dbs}")

    lines.append(f"\n{'-' * 60}")
    lines.append("3. INTERPRETACION")
    lines.append("-" * 60)
    lines.append("  - Las monedas en 'Coincidencias' son anomalías robustas:")
    lines.append("    dos métodos con lógicas distintas las señalan consistentemente.")
    lines.append("  - Las monedas 'Solo en SVM' merecen revisión: SVM detecta fronteras")
    lines.append("    en el espacio de features que DBSCAN no captura por densidad.")
    lines.append("  - Las monedas 'Solo en DBSCAN' son outliers de densidad local,")
    lines.append("    no necesariamente outliers globales en el espacio de features.")

    lines.append(f"\n{'-' * 60}")
    lines.append("4. RECOMENDACIONES PARA LA FASE SUPERVISADA")
    lines.append("-" * 60)
    lines.append(f"  - Excluir o tratar separadamente: {coinciden}")
    lines.append("  - Usar 'OneClassSVM_consensus' de cluster_labels.csv como feature binaria.")
    lines.append("  - Priorizar BTC y ETH como targets si están en el grupo anómalo:")
    lines.append("    su comportamiento único los hace más predecibles con modelos propios.")
    lines.append("=" * 60)

    out = REPORTS_DIR / "svm_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out}")


def update_labels(labels_df: pd.DataFrame, symbols: list,
                  results: dict, consensus: np.ndarray) -> None:
    """
    Agrega dos columnas a cluster_labels.csv con la misma convención que DBSCAN:
      -1 = anomalía, 1 = normal
    - OneClassSVM_nu020: predicción puntual con nu=0.20
    - OneClassSVM_consensus: anomalía estable en ≥3 nu values
    """
    labels_df["OneClassSVM_nu020"]     = pd.Series(results[0.20], index=symbols)
    labels_df["OneClassSVM_consensus"] = pd.Series(consensus,     index=symbols)
    labels_df.to_csv(CSV_LABELS)
    print(f"[save] {CSV_LABELS} — columnas OneClassSVM_nu020, OneClassSVM_consensus agregadas")
    print(labels_df[["DBSCAN", "OneClassSVM_nu020", "OneClassSVM_consensus"]].to_string())


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
    plot_vs_dbscan(X, consensus, labels["DBSCAN"].values, symbols)

    print("\n--- Conclusiones ---")
    save_conclusions(results, consensus, labels["DBSCAN"].values, symbols)

    print("\n--- Actualizando cluster_labels.csv ---")
    update_labels(labels, symbols, results, consensus)

    print("\n[done] svm_analysis completado.")


if __name__ == "__main__":
    main()

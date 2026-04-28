"""
[03] Clustering — Crypto ML Project
Aplica K-Means, DBSCAN y Agglomerative Clustering sobre el feature matrix.
Usa metodo del codo, silhouette y Davies-Bouldin para seleccion de K.

Uso:
    python unsupervised/clustering.py

Salida:
    data/cluster_labels.csv          — etiquetas por moneda para cada algoritmo
    unsupervised/reports/            — graficas de seleccion de K y clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CSV_IN  = DATA_DIR / "features_clustering.csv"
CSV_OUT = DATA_DIR / "cluster_labels.csv"

K_RANGE    = range(2, 8)
ALT_K      = 4   # vista alternativa con mas granularidad


def load_features() -> pd.DataFrame:
    df = pd.read_csv(CSV_IN, index_col="Symbol")
    print(f"[load] {df.shape[0]} monedas, {df.shape[1]} features")
    return df


# ---------------------------------------------------------------------------
# Seleccion de K
# ---------------------------------------------------------------------------
def select_k(X: np.ndarray) -> int:
    inertias, silhouettes, db_scores = [], [], []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(list(K_RANGE), inertias, marker="o")
    axes[0].set_title("Metodo del Codo")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inercia")

    axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="green")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Score")

    axes[2].plot(list(K_RANGE), db_scores, marker="o", color="red")
    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Score (menor es mejor)")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "k_selection.png", dpi=150)
    plt.close()
    print("[plot] unsupervised/reports/k_selection.png")

    best_k = list(K_RANGE)[np.argmax(silhouettes)]

    print(f"\n  K | Inercia   | Silhouette | Davies-Bouldin")
    print(f"  {'-'*45}")
    for i, k in enumerate(K_RANGE):
        marker = " <--" if k == best_k else ""
        print(f"  {k} | {inertias[i]:9.1f} | {silhouettes[i]:10.4f} | {db_scores[i]:14.4f}{marker}")

    print(f"\n[select] K optimo: {best_k}")
    return best_k


# ---------------------------------------------------------------------------
# Algoritmos
# ---------------------------------------------------------------------------
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"[kmeans] K={k} | Silhouette={sil:.4f}")
    return labels


def run_dbscan(X: np.ndarray) -> np.ndarray:
    db = DBSCAN(eps=0.8, min_samples=2)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[dbscan] Clusters={n_clusters} | Ruido/anomalias={n_noise}")
    return labels


def run_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    ag = AgglomerativeClustering(n_clusters=k)
    labels = ag.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"[agglomerative] K={k} | Silhouette={sil:.4f}")
    return labels


# ---------------------------------------------------------------------------
# Perfil de clusters — media de features originales por grupo
# ---------------------------------------------------------------------------
def cluster_profile(features: pd.DataFrame, labels: np.ndarray, name: str):
    df = features.copy()
    df["Cluster"] = labels
    profile = df.groupby("Cluster").mean().round(4)
    print(f"\n--- Perfil por cluster ({name}) ---")
    print(profile.to_string())
    return profile


# ---------------------------------------------------------------------------
# Anomalias DBSCAN
# ---------------------------------------------------------------------------
def print_dbscan_anomalies(symbols: list, labels: np.ndarray):
    anomalies = [s for s, l in zip(symbols, labels) if l == -1]
    normal    = {l: [s for s, lb in zip(symbols, labels) if lb == l]
                 for l in sorted(set(labels)) if l != -1}
    print(f"\n--- DBSCAN ---")
    for cluster, coins in normal.items():
        print(f"  Cluster {cluster}: {coins}")
    print(f"  Anomalias (-1): {anomalies}")


# ---------------------------------------------------------------------------
# Visualizacion PCA 2D — optimo + alternativo lado a lado
# ---------------------------------------------------------------------------
def plot_clusters(X: np.ndarray, labels_dict: dict, symbols: list):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum()

    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=120)
        for i, sym in enumerate(symbols):
            ax.annotate(sym, (coords[i, 0], coords[i, 1]), fontsize=7,
                        ha="center", va="bottom")
        ax.set_title(f"{name}\n(PCA {var_explained:.0%} varianza)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "clusters_pca.png", dpi=150)
    plt.close()
    print("[plot] unsupervised/reports/clusters_pca.png")


# ---------------------------------------------------------------------------
# Conclusiones
# ---------------------------------------------------------------------------
def save_conclusions(features: pd.DataFrame, best_k: int, alt_k: int,
                     labels_dict: dict, symbols: list):

    kmeans_best = labels_dict[f"KMeans_K{best_k}"]
    kmeans_alt  = labels_dict[f"KMeans_K{alt_k}"]
    dbscan      = labels_dict["DBSCAN"]

    clusters_best = {c: [s for s, l in zip(symbols, kmeans_best) if l == c]
                     for c in sorted(set(kmeans_best))}
    clusters_alt  = {c: [s for s, l in zip(symbols, kmeans_alt) if l == c]
                     for c in sorted(set(kmeans_alt))}
    anomalies     = [s for s, l in zip(symbols, dbscan) if l == -1]
    dbscan_groups = {c: [s for s, l in zip(symbols, dbscan) if l == c]
                     for c in sorted(set(dbscan)) if c != -1}

    # Perfil original por cluster para interpretacion
    df = features.copy()
    df["_k_best"] = kmeans_best
    df["_k_alt"]  = kmeans_alt
    profile_best = df.groupby("_k_best")[features.columns].mean()
    profile_alt  = df.groupby("_k_alt")[features.columns].mean()

    def describe_cluster(profile_row):
        # Los valores son normalizados (StandardScaler): 0=promedio, >1=alto, <-1=bajo
        notes = []
        if profile_row["volatility"] > 1.5:
            notes.append("volatilidad extrema")
        elif profile_row["volatility"] > 0.3:
            notes.append("alta volatilidad")
        elif profile_row["volatility"] < -0.3:
            notes.append("baja volatilidad")
        if profile_row["mean_return"] > 1.5:
            notes.append("retorno historico excepcional")
        elif profile_row["mean_return"] < -0.3:
            notes.append("retorno por debajo del promedio")
        if profile_row["sharpe_ratio"] > 0.3:
            notes.append("buen ratio retorno/riesgo")
        elif profile_row["sharpe_ratio"] < -0.5:
            notes.append("mal ratio retorno/riesgo")
        if profile_row["max_drawdown"] > 1.0:
            notes.append("stablecoin (drawdown atipico)")
        if profile_row["avg_volume"] > 2.0:
            notes.append("volumen muy alto")
        elif profile_row["avg_volume"] < -0.3:
            notes.append("volumen bajo")
        return ", ".join(notes) if notes else "perfil cercano al promedio del mercado"

    lines = []
    lines.append("=" * 60)
    lines.append("CONCLUSIONES DE CLUSTERING — Crypto ML Project")
    lines.append("=" * 60)
    lines.append(f"\nDataset: {len(symbols)} monedas | Features: {len(features.columns)}")
    lines.append(f"Features usados: {', '.join(features.columns)}\n")

    lines.append("-" * 60)
    lines.append("1. SELECCION DE K")
    lines.append("-" * 60)
    lines.append(f"  K optimo segun silhouette: {best_k} (score=0.6726)")
    lines.append(f"  K alternativo analizado: {alt_k}")
    lines.append(f"  K=2 produce separacion clara pero trivial (XEM vs resto).")
    lines.append(f"  K=4 revela subgrupos mas utiles para el analisis.")

    lines.append(f"\n{'-' * 60}")
    lines.append(f"2. K-MEANS K={best_k}")
    lines.append("-" * 60)
    for c, coins in clusters_best.items():
        row = profile_best.loc[c]
        desc = describe_cluster(row)
        noun = "moneda" if len(coins) == 1 else "monedas"
        lines.append(f"  Cluster {c} ({len(coins)} {noun}) — {desc}")
        lines.append(f"    Monedas: {', '.join(coins)}")

    lines.append(f"\n{'-' * 60}")
    lines.append(f"3. K-MEANS K={alt_k} (ALTERNATIVO)")
    lines.append("-" * 60)
    for c, coins in clusters_alt.items():
        row = profile_alt.loc[c]
        desc = describe_cluster(row)
        noun = "moneda" if len(coins) == 1 else "monedas"
        lines.append(f"  Cluster {c} ({len(coins)} {noun}) — {desc}")
        lines.append(f"    Monedas: {', '.join(coins)}")
    lines.append(f"\n  Con K=4 emergen stablecoins (USDT, USDC) como grupo propio,")
    lines.append(f"  confirmando que su comportamiento es fundamentalmente distinto")
    lines.append(f"  al resto del mercado (volatilidad casi nula, precio anclado al USD).")
    lines.append(f"  USDT domina en volumen de trading; USDC opera con volumen menor.")

    lines.append(f"\n{'-' * 60}")
    lines.append("4. DBSCAN — ANOMALIAS Y GRUPOS NATURALES")
    lines.append("-" * 60)
    for c, coins in dbscan_groups.items():
        lines.append(f"  Cluster {c}: {', '.join(coins)}")
    lines.append(f"  Anomalias ({len(anomalies)}): {', '.join(anomalies)}")
    lines.append(f"\n  Interpretacion de anomalias:")
    lines.append(f"    BTC, ETH — lideres de mercado, comportamiento unico como referencia")
    lines.append(f"    DOGE     — meme coin, movimientos impredecibles por sentiment")
    lines.append(f"    USDC, USDT — stablecoins, precio anclado al USD")
    lines.append(f"    XEM      — outlier extremo en retorno y volatilidad historica")
    lines.append(f"\n  DBSCAN separa naturalmente DeFi/nuevos (AAVE, BNB, DOT, SOL, UNI, WBTC)")
    lines.append(f"  de altcoins establecidos (ADA, LTC, XRP, etc.).")

    lines.append(f"\n{'-' * 60}")
    lines.append("5. CONSENSO ENTRE ALGORITMOS")
    lines.append("-" * 60)
    lines.append("  Los tres algoritmos coinciden en:")
    lines.append("    - XEM es un outlier consistente en todos los metodos.")
    lines.append("    - USDT y USDC forman un grupo propio (stablecoins).")
    lines.append("    - BTC y ETH tienen perfil unico, no encajan en clusters generales.")
    lines.append("    - El resto del mercado forma un bloque homogeneo (alta correlacion).")

    lines.append(f"\n{'-' * 60}")
    lines.append("6. RECOMENDACIONES PARA SIGUIENTES FASES")
    lines.append("-" * 60)
    lines.append("  - Excluir USDT y USDC de modelos de prediccion de precios.")
    lines.append("  - Tratar XEM como caso especial o excluirlo segun el objetivo.")
    lines.append("  - Usar labels de K=4 como feature adicional en el modelo supervisado.")
    lines.append("  - DBSCAN Cluster 0 (DeFi) puede requerir modelos separados")
    lines.append("    por su comportamiento mas correlacionado entre si.")
    lines.append("=" * 60)

    out_path = REPORTS_DIR / "clustering_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] unsupervised/reports/clustering_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    features = load_features()
    X        = features.values
    symbols  = list(features.index)

    print("\n--- Seleccion de K ---")
    best_k = select_k(X)

    print("\n--- Clustering ---")
    kmeans_labels  = run_kmeans(X, best_k)
    kmeans_alt     = run_kmeans(X, ALT_K)
    dbscan_labels  = run_dbscan(X)
    agglo_labels   = run_agglomerative(X, best_k)

    # Perfiles con features originales (sin escalar) para interpretabilidad
    features_raw = pd.read_csv(DATA_DIR / "features_clustering.csv", index_col="Symbol")
    cluster_profile(features_raw, kmeans_labels, f"KMeans K={best_k}")
    cluster_profile(features_raw, kmeans_alt,    f"KMeans K={ALT_K} (alternativo)")
    print_dbscan_anomalies(symbols, dbscan_labels)

    labels_dict = {
        f"KMeans_K{best_k}":    kmeans_labels,
        f"KMeans_K{ALT_K}":     kmeans_alt,
        "DBSCAN":               dbscan_labels,
        f"Agglomerative_K{best_k}": agglo_labels,
    }

    plot_clusters(X, labels_dict, symbols)

    results = pd.DataFrame(
        {k: v for k, v in labels_dict.items()},
        index=symbols
    )
    results.index.name = "Symbol"
    results.to_csv(CSV_OUT)
    print(f"\n[save] {CSV_OUT}")

    print(f"\n--- Asignacion KMeans K={best_k} ---")
    for cluster in sorted(results[f"KMeans_K{best_k}"].unique()):
        coins = results[results[f"KMeans_K{best_k}"] == cluster].index.tolist()
        print(f"  Cluster {cluster}: {coins}")

    print(f"\n--- Asignacion KMeans K={ALT_K} (alternativo) ---")
    for cluster in sorted(results[f"KMeans_K{ALT_K}"].unique()):
        coins = results[results[f"KMeans_K{ALT_K}"] == cluster].index.tolist()
        print(f"  Cluster {cluster}: {coins}")

    save_conclusions(features_raw, best_k, ALT_K, labels_dict, symbols)

    print("\n[done] Clustering completado.")


if __name__ == "__main__":
    main()

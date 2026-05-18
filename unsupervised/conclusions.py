"""
[03d] Conclusiones Unsupervised — Crypto ML Project
Analisis global y conclusivo de todos los algoritmos de clustering + SVM.

Uso:
    python unsupervised/conclusions.py

Salida:
    unsupervised/reports/conclusions_unsupervised.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

CSV_LABELS   = DATA_DIR / "cluster_labels.csv"
CSV_FEATURES = DATA_DIR / "features_clustering.csv"


def load_data() -> tuple:
    labels   = pd.read_csv(CSV_LABELS,   index_col="Symbol")
    features = pd.read_csv(CSV_FEATURES, index_col="Symbol")
    print(f"[load] {labels.shape[0]} monedas | columnas: {list(labels.columns)}")
    return labels, features


# ---------------------------------------------------------------------------
# Tabla cruzada de anomalias
# ---------------------------------------------------------------------------
def anomaly_cross_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla booleana: fila=moneda, col=algoritmo, True=anomalia."""
    cross = pd.DataFrame(index=labels.index)
    cross["DBSCAN"]           = labels["DBSCAN"] == -1
    cross["OneClassSVM"]      = labels["OneClassSVM_consensus"] == -1
    cross["KMeans_K2_outlier"] = labels["KMeans_K2"] == 1          # XEM solo
    cross["Agglom_outlier"]    = labels["Agglomerative_K2"] == 1   # XEM solo
    cross["n_algorithms"]      = cross.sum(axis=1)
    return cross


def describe_coin_profile(sym: str, features: pd.DataFrame) -> str:
    """Retorna descripcion breve del perfil de una moneda en features originales."""
    row = features.loc[sym]
    notes = []
    if row["volatility"] > features["volatility"].quantile(0.75):
        notes.append("alta volatilidad")
    elif row["volatility"] < features["volatility"].quantile(0.25):
        notes.append("baja volatilidad")
    if row["mean_return"] > features["mean_return"].quantile(0.75):
        notes.append("alto retorno historico")
    elif row["mean_return"] < features["mean_return"].quantile(0.25):
        notes.append("bajo retorno historico")
    if row["avg_volume"] > features["avg_volume"].quantile(0.75):
        notes.append("volumen alto")
    elif row["avg_volume"] < features["avg_volume"].quantile(0.25):
        notes.append("volumen bajo")
    if row["max_drawdown"] > -0.5:
        notes.append("drawdown atipico (posible stablecoin)")
    return ", ".join(notes) if notes else "perfil cercano al promedio"


# ---------------------------------------------------------------------------
# Grupos estables DBSCAN
# ---------------------------------------------------------------------------
def dbscan_groups(labels: pd.DataFrame) -> dict:
    groups = {}
    for cluster in sorted(labels["DBSCAN"].unique()):
        coins = labels[labels["DBSCAN"] == cluster].index.tolist()
        groups[cluster] = coins
    return groups


# ---------------------------------------------------------------------------
# Generacion del reporte
# ---------------------------------------------------------------------------
def generate_report(labels: pd.DataFrame, features: pd.DataFrame) -> str:
    cross   = anomaly_cross_table(labels)
    symbols = list(labels.index)
    lines   = []

    lines.append("=" * 65)
    lines.append("CONCLUSIONES UNSUPERVISED — Crypto ML Project")
    lines.append("=" * 65)
    n_real  = sum(1 for s in symbols if not s.startswith("SYN"))
    n_synth = len(symbols) - n_real
    lines.append(f"\nDataset : {len(symbols)} monedas ({n_real} reales + {n_synth} sintéticas) | 2013-2024")
    lines.append(f"Features: mean_return, volatility, sharpe_ratio, max_drawdown, avg_volume")
    lines.append(f"Algoritmos: K-Means (K=2, K=4), DBSCAN, Agglomerative (K=2), One-Class SVM\n")

    # ---- 1. Tabla cruzada de anomalias ----
    lines.append("-" * 65)
    lines.append("1. ANOMALIAS POR ALGORITMO")
    lines.append("-" * 65)
    header = f"{'Moneda':>8} | {'DBSCAN':>6} | {'SVM':>5} | {'KMeans':>6} | {'Agglom':>6} | Total"
    lines.append(header)
    lines.append("-" * len(header))
    for sym in symbols:
        r = cross.loc[sym]
        dbscan = "  ANOM" if r["DBSCAN"]           else "normal"
        svm    = " ANOM" if r["OneClassSVM"]       else "  ok "
        km     = "  ANOM" if r["KMeans_K2_outlier"] else "normal"
        ag     = "  ANOM" if r["Agglom_outlier"]    else "normal"
        total  = int(r["n_algorithms"])
        marker = " <-- ROBUSTO" if total >= 3 else (" <-- revisar" if total == 2 else "")
        lines.append(f"{sym:>8} | {dbscan} | {svm} | {km} | {ag} | {total}/4{marker}")

    # ---- 2. Consenso global ----
    lines.append(f"\n{'-' * 65}")
    lines.append("2. CONSENSO GLOBAL DE ANOMALIAS")
    lines.append("-" * 65)

    for n in [4, 3, 2, 1]:
        coins = cross[cross["n_algorithms"] == n].index.tolist()
        if not coins:
            continue
        label = "ANOMALIA ROBUSTA (≥3 algoritmos)" if n >= 3 else f"Señalada por {n} algoritmo(s)"
        lines.append(f"\n  {label}:")
        for sym in coins:
            profile = describe_coin_profile(sym, features)
            lines.append(f"    {sym:>6}: {profile}")

    normal = cross[cross["n_algorithms"] == 0].index.tolist()
    lines.append(f"\n  Normal en todos los algoritmos ({len(normal)}):")
    lines.append(f"    {', '.join(normal)}")

    # ---- 3. Grupos estables DBSCAN ----
    lines.append(f"\n{'-' * 65}")
    lines.append("3. SEGMENTACION DEL MERCADO (DBSCAN)")
    lines.append("-" * 65)
    groups = dbscan_groups(labels)
    group_labels = {-1: "Anomalias / comportamiento unico"}
    for cluster, coins in groups.items():
        name = group_labels.get(cluster, f"Cluster {cluster}")
        real_coins  = [c for c in coins if not c.startswith("SYN")]
        synth_count = len(coins) - len(real_coins)
        lines.append(f"\n  {name} ({len(coins)} total — {len(real_coins)} reales, {synth_count} sintéticas):")
        lines.append(f"    Reales: {', '.join(real_coins) if real_coins else '(ninguna)'}")

    # ---- 4. Perfil de clusters KMeans K=4 ----
    lines.append(f"\n{'-' * 65}")
    lines.append("4. PERFIL DE GRUPOS — K-MEANS K=4")
    lines.append("-" * 65)
    for cluster in sorted(labels["KMeans_K4"].unique()):
        coins = labels[labels["KMeans_K4"] == cluster].index.tolist()
        real_coins  = [c for c in coins if not c.startswith("SYN")]
        synth_count = len(coins) - len(real_coins)
        group_features = features.loc[coins].mean()
        lines.append(f"\n  Cluster {cluster} ({len(coins)} total — {len(real_coins)} reales, {synth_count} sintéticas):")
        lines.append(f"    Reales     : {', '.join(real_coins) if real_coins else '(ninguna)'}")
        lines.append(f"    Volatilidad: {group_features['volatility']:.6f} (media del grupo)")
        lines.append(f"    Retorno    : {group_features['mean_return']:.6f}")
        lines.append(f"    Sharpe     : {group_features['sharpe_ratio']:.4f}")
        lines.append(f"    Max DD     : {group_features['max_drawdown']:.4f}")

    # ---- 5. Llamado a la accion ----
    lines.append(f"\n{'-' * 65}")
    lines.append("5. LLAMADO A LA ACCION — FASE SUPERVISADA")
    lines.append("-" * 65)

    robust_anomalies = [s for s in cross[cross["n_algorithms"] >= 3].index.tolist() if not s.startswith("SYN")]
    defi_group       = [c for c in groups.get(0, []) if not c.startswith("SYN")]
    altcoin_group    = [c for c in groups.get(1, []) if not c.startswith("SYN")]

    lines.append(f"""
  A. EXCLUSIONES RECOMENDADAS
     Anomalias robustas (≥3 algoritmos): {robust_anomalies}
     → Excluir USDT y USDC como targets de prediccion de precio.
     → Excluir XEM del modelo general; analizarlo por separado si el objetivo
       es detectar eventos extremos de mercado.
     → SOL y WBTC: señaladas solo por SVM. No excluir, pero monitorear su
       comportamiento en validacion del modelo supervisado.

  B. FEATURES ADICIONALES RECOMENDADAS
     → Incluir 'KMeans_K4' como feature categorica (4 grupos interpretables).
     → Incluir 'OneClassSVM_consensus' como feature binaria de anomalia global.
     → Incluir 'DBSCAN' (binarizado: -1 vs normal) para capturar outliers de densidad.

  C. SEGMENTACION PARA MODELADO
     → Grupo DeFi ({', '.join(defi_group)}):
        Alta correlacion interna. Considerar modelo separado o feature de grupo.
     → Altcoins establecidos ({', '.join(altcoin_group)}):
        Comportamiento homogeneo. Pueden modelarse conjuntamente.
     → BTC y ETH: anomalias de densidad (DBSCAN) pero no de features globales (SVM).
        Son los mejores candidatos como targets principales del modelo supervisado
        por su historia larga y menor ruido idiosincratico.

  D. SPLIT TEMPORAL
     → 80 / 10 / 10 estrictamente cronologico. No mezclar fechas entre splits.
     → Priorizar BTC y ETH (mayor cobertura historica desde 2013-2015).
     → Evitar AAVE, DOT, SOL, UNI (datos solo desde 2020, <2 años).
""")
    lines.append("=" * 65)
    return "\n".join(lines)


def main():
    labels, features = load_data()

    print("\n--- Generando conclusiones unsupervised ---")
    report = generate_report(labels, features)

    out = REPORTS_DIR / "conclusions_unsupervised.txt"
    out.write_text(report, encoding="utf-8")
    print(f"[save] {out}")
    print("\n" + report)
    print("\n[done] Conclusiones completadas.")


if __name__ == "__main__":
    main()

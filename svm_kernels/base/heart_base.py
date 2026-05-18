#!/usr/bin/env python3
"""
Baseline: Heart Disease + Extremality MKL
Kernels débiles polinomiales con ponderación por extremalidad.
Compara Natural MKL, Anti-Natural MKL, RBF y Polynomial (d=3).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weak_polinomial_kernel import create_weak_kernels
from src.kernel_metrics import kernel_aligment, kernel_polarization, FSM, complex_ratio
from extremalitymkl.extremality_weights import kernel_extremaly_weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

# ── Dataset ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "heart.csv"))
X_raw = np.array(data.iloc[:, :-1], dtype=float)
y_raw = np.array(data.iloc[:, -1])
y = np.where(y_raw == 0, -1, y_raw)

# ── Parámetros ────────────────────────────────────────────────────────────────
ITERATIONS   = 30
NUM_KERNELS  = 20
MAX_FEATURES = 5
N_WEIGHT     = 2

METHODS = ['Natural MKL', 'Anti-Natural MKL', 'RBF', 'Polynomial (d=3)']
COLORS  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
QUALITY_LABELS = ['Alignment', 'Polarization', 'FSM', 'Complex Ratio']
SCORE_LABELS   = ['Accuracy', 'F1', 'AUC']

# ── Funciones ─────────────────────────────────────────────────────────────────
def kernel_quality(K, y):
    return np.array([
        kernel_aligment(K, y),
        kernel_polarization(K, y),
        FSM(K, y),
        complex_ratio(K),
    ])

def svm_scores(K_train, K_test, y_train, y_test, C=1.0):
    clf = SVC(kernel='precomputed', C=C, probability=True)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    y_prob = clf.predict_proba(K_test)[:, 1]
    return np.array([
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, pos_label=1),
        roc_auc_score((y_test == 1).astype(int), y_prob),
    ])

def one_fold(X_train, X_test, y_train, y_test):
    KL_train, KL_test = create_weak_kernels(
        X_train, X_test, t=MAX_FEATURES, num_kernels=NUM_KERNELS
    )
    weights = kernel_extremaly_weights(KL_train, y_train, n=N_WEIGHT)

    K_nat_tr  = np.einsum('ijk,i->jk', KL_train, weights.w_1)
    K_nat_te  = np.einsum('ijk,i->jk', KL_test,  weights.w_1)
    K_anti_tr = np.einsum('ijk,i->jk', KL_train, weights.w_2)
    K_anti_te = np.einsum('ijk,i->jk', KL_test,  weights.w_2)
    K_rbf_tr  = rbf_kernel(X_train)
    K_rbf_te  = rbf_kernel(X_test, X_train)
    K_poly_tr = polynomial_kernel(X_train, degree=3)
    K_poly_te = polynomial_kernel(X_test, X_train, degree=3)

    kernels = {
        'Natural MKL':      (K_nat_tr,  K_nat_te),
        'Anti-Natural MKL': (K_anti_tr, K_anti_te),
        'RBF':              (K_rbf_tr,  K_rbf_te),
        'Polynomial (d=3)': (K_poly_tr, K_poly_te),
    }
    quality, scores = {}, {}
    for name, (Ktr, Kte) in kernels.items():
        quality[name] = kernel_quality(Ktr, y_train)
        scores[name]  = svm_scores(Ktr, Kte, y_train, y_test)
    return quality, scores

def run_simulation(x_raw, y, iterations):
    x = MinMaxScaler().fit_transform(x_raw)
    all_q = {m: [] for m in METHODS}
    all_s = {m: [] for m in METHODS}
    for k in range(iterations):
        X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=k)
        q, s = one_fold(X_tr, X_te, y_tr, y_te)
        for m in METHODS:
            all_q[m].append(q[m])
            all_s[m].append(s[m])
    mean_q = {m: np.mean(all_q[m], axis=0) for m in METHODS}
    std_q  = {m: np.std(all_q[m],  axis=0) for m in METHODS}
    mean_s = {m: np.mean(all_s[m],  axis=0) for m in METHODS}
    std_s  = {m: np.std(all_s[m],   axis=0) for m in METHODS}
    return mean_q, std_q, mean_s, std_s

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Baseline — {ITERATIONS} iteraciones, {NUM_KERNELS} kernels (poly), "
          f"max_features={MAX_FEATURES}")
    mq, sq, ms, ss = run_simulation(X_raw, y, ITERATIONS)

    print("\n── Kernel Quality (media ± std) " + "─" * 40)
    print(f"{'Método':<22}" + "".join(f"{n:>20}" for n in QUALITY_LABELS))
    for m in METHODS:
        print(f"{m:<22}" + "".join(f"{mq[m][i]:>10.4f} ±{sq[m][i]:.4f}" for i in range(4)))

    print("\n── Clasificación SVM (media ± std) " + "─" * 36)
    print(f"{'Método':<22}" + "".join(f"{n:>20}" for n in SCORE_LABELS))
    for m in METHODS:
        print(f"{m:<22}" + "".join(f"{ms[m][i]:>10.4f} ±{ss[m][i]:.4f}" for i in range(3)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # Plot clasificación
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Baseline SVM — Heart Disease ({ITERATIONS} iters)", fontsize=13)
    for i, (label, ax) in enumerate(zip(SCORE_LABELS, axes)):
        bars = ax.bar(METHODS, [ms[m][i] for m in METHODS],
                      yerr=[ss[m][i] for m in METHODS],
                      capsize=5, color=COLORS, alpha=0.85, ecolor='gray')
        ax.set_title(label); ax.set_ylim(0, 1.12)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=22, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, m in zip(bars, METHODS):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{ms[m][i]:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "heart_classification.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot calidad de kernel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Baseline Kernel Quality — Heart Disease", fontsize=13)
    for i, (label, ax) in enumerate(zip(QUALITY_LABELS, axes.flat)):
        ax.bar(METHODS, [mq[m][i] for m in METHODS],
               yerr=[sq[m][i] for m in METHODS],
               capsize=5, color=COLORS, alpha=0.85, ecolor='gray')
        ax.set_title(label)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=22, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(out, "heart_kernel_quality.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Plots guardados en base/results/")

    # Exportar resultados como numpy para usar en comparison.py
    np.save(os.path.join(out, "mean_scores.npy"), {m: ms[m] for m in METHODS})
    np.save(os.path.join(out, "std_scores.npy"),  {m: ss[m] for m in METHODS})

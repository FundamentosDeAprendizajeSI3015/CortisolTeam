#!/usr/bin/env python3
"""
Versión mejorada: Heart Disease + Extremality MKL
Mejoras sobre el baseline:
  1. Pool de kernels diversificado: polinomiales + RBF multi-gamma + Laplacian
  2. Búsqueda de C óptimo por validación cruzada interna (3-fold)
  3. Ponderación con las 4 métricas (alignment, polarization, FSM, complex_ratio)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kernel_metrics import kernel_aligment, kernel_polarization, FSM, complex_ratio
from extremalitymkl.extremality_weights import kernel_extremaly_weights, metrics_kernels
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel

# ── Dataset ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "heart.csv"))
X_raw = np.array(data.iloc[:, :-1], dtype=float)
y_raw = np.array(data.iloc[:, -1])
y = np.where(y_raw == 0, -1, y_raw)

# ── Parámetros ────────────────────────────────────────────────────────────────
ITERATIONS   = 30
NUM_KERNELS  = 20   # kernels por tipo en el pool
MAX_FEATURES = 5
N_WEIGHT     = 2
C_GRID       = [0.1, 1, 10, 100]

# Gammas para RBF/Laplacian (escala respecto a 1/n_features)
N_FEATURES = X_raw.shape[1]
GAMMAS = [0.01 / N_FEATURES, 0.1 / N_FEATURES, 1.0 / N_FEATURES, 10.0 / N_FEATURES]
POLY_DEGREES = [2, 3, 4]

METHODS = ['Natural MKL+', 'Anti-Natural MKL+', 'RBF (tuned)', 'Polynomial (tuned)']
COLORS  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
QUALITY_LABELS = ['Alignment', 'Polarization', 'FSM', 'Complex Ratio']
SCORE_LABELS   = ['Accuracy', 'F1', 'AUC']

# ── Mejora 1: Pool de kernels diversificado ───────────────────────────────────
def create_diverse_kernels(X_train, X_test, num_kernels, max_features):
    """
    Genera un pool mixto: kernels polinomiales (features aleatorias),
    RBF con distintos gamma y Laplacian con distintos gamma.
    """
    KL_train, KL_test = [], []
    rng = np.random.default_rng(seed=None)
    n_cols = X_train.shape[1]

    for _ in range(num_kernels):
        kernel_type = rng.choice(['poly', 'rbf', 'laplacian'])
        cols = rng.integers(0, n_cols, size=rng.integers(1, max_features + 1))
        X1 = X_train[:, cols]
        X2 = X_test[:, cols]

        if kernel_type == 'poly':
            d = int(rng.choice(POLY_DEGREES))
            Ktr = polynomial_kernel(X1, degree=d, coef0=0, gamma=1)
            Kte = polynomial_kernel(X2, X1, degree=d, coef0=0, gamma=1)
        elif kernel_type == 'rbf':
            g = float(rng.choice(GAMMAS))
            Ktr = rbf_kernel(X1, gamma=g)
            Kte = rbf_kernel(X2, X1, gamma=g)
        else:  # laplacian
            g = float(rng.choice(GAMMAS))
            Ktr = laplacian_kernel(X1, gamma=g)
            Kte = laplacian_kernel(X2, X1, gamma=g)

        KL_train.append(Ktr)
        KL_test.append(Kte)

    return np.array(KL_train), np.array(KL_test)

# ── Mejora 2: C óptimo por CV interna ────────────────────────────────────────
def best_C(K_train, y_train, C_grid, n_splits=3):
    """Elige C por accuracy media en CV interna sobre el kernel de entrenamiento."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    best, best_score = C_grid[0], -1.0
    for C in C_grid:
        scores = []
        for tr, va in skf.split(K_train, y_train):
            clf = SVC(kernel='precomputed', C=C)
            clf.fit(K_train[np.ix_(tr, tr)], y_train[tr])
            scores.append(accuracy_score(y_train[va], clf.predict(K_train[np.ix_(va, tr)])))
        mean = np.mean(scores)
        if mean > best_score:
            best_score, best = mean, C
    return best

# ── Métricas de calidad y clasificación ──────────────────────────────────────
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

# ── Mejora 3: ponderación con las 4 métricas ─────────────────────────────────
from src.kernel_metrics import kernel_aligment, kernel_polarization, FSM, complex_ratio
from extremalitymkl.extremality_weights import KernelWeights
from extremalitymkl.extremality_order import order_compar
from src.weigth_linear_combination import weight

ALL_METRICS = {
    "alignment":    (lambda K, y: kernel_aligment(K, y),     1),
    "polarization": (lambda K, y: kernel_polarization(K, y), 1),
    "FSM":          (lambda K, y: FSM(K, y),                 1),
    "complex_ratio":(lambda K, y: complex_ratio(K),          -1),
}

def kernel_extremaly_weights_all(KL_train, y_train, n=2):
    """Extremality weights usando las 4 métricas disponibles."""
    num_k = KL_train.shape[0]
    measures = np.zeros((num_k, len(ALL_METRICS)))
    directions = np.zeros(len(ALL_METRICS))
    for j, (name, (fn, direction)) in enumerate(ALL_METRICS.items()):
        for i in range(num_k):
            measures[i, j] = fn(KL_train[i], y_train)
        directions[j] = direction

    o1 = order_compar(measures, directions)
    w1 = weight(len(o1) - o1, n)
    o2 = order_compar(measures, -1 * directions)
    w2 = weight(o2, n)
    return KernelWeights(w1, w2)

# ── Un fold completo ──────────────────────────────────────────────────────────
def one_fold(X_train, X_test, y_train, y_test):
    # Pool diversificado
    KL_train, KL_test = create_diverse_kernels(
        X_train, X_test, NUM_KERNELS, MAX_FEATURES
    )

    # Ponderación con 4 métricas
    weights = kernel_extremaly_weights_all(KL_train, y_train, n=N_WEIGHT)

    K_nat_tr  = np.einsum('ijk,i->jk', KL_train, weights.w_1)
    K_nat_te  = np.einsum('ijk,i->jk', KL_test,  weights.w_1)
    K_anti_tr = np.einsum('ijk,i->jk', KL_train, weights.w_2)
    K_anti_te = np.einsum('ijk,i->jk', KL_test,  weights.w_2)

    # Baselines con gamma óptimo elegido del grid
    best_gamma = GAMMAS[2]  # 1/n_features como default; el grid se aplica al SVM
    K_rbf_tr  = rbf_kernel(X_train, gamma=best_gamma)
    K_rbf_te  = rbf_kernel(X_test, X_train, gamma=best_gamma)
    K_poly_tr = polynomial_kernel(X_train, degree=3)
    K_poly_te = polynomial_kernel(X_test, X_train, degree=3)

    kernels = {
        'Natural MKL+':      (K_nat_tr,  K_nat_te),
        'Anti-Natural MKL+': (K_anti_tr, K_anti_te),
        'RBF (tuned)':       (K_rbf_tr,  K_rbf_te),
        'Polynomial (tuned)':(K_poly_tr, K_poly_te),
    }

    quality, scores = {}, {}
    for name, (Ktr, Kte) in kernels.items():
        C = best_C(Ktr, y_train, C_GRID)
        quality[name] = kernel_quality(Ktr, y_train)
        scores[name]  = svm_scores(Ktr, Kte, y_train, y_test, C=C)
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
    print(f"Mejorado — {ITERATIONS} iteraciones, {NUM_KERNELS} kernels (poly+RBF+Laplacian), "
          f"max_features={MAX_FEATURES}, C-grid={C_GRID}")
    mq, sq, ms, ss = run_simulation(X_raw, y, ITERATIONS)

    print("\n── Kernel Quality (media ± std) " + "─" * 40)
    print(f"{'Método':<24}" + "".join(f"{n:>20}" for n in QUALITY_LABELS))
    for m in METHODS:
        print(f"{m:<24}" + "".join(f"{mq[m][i]:>10.4f} ±{sq[m][i]:.4f}" for i in range(4)))

    print("\n── Clasificación SVM (media ± std) " + "─" * 36)
    print(f"{'Método':<24}" + "".join(f"{n:>20}" for n in SCORE_LABELS))
    for m in METHODS:
        print(f"{m:<24}" + "".join(f"{ms[m][i]:>10.4f} ±{ss[m][i]:.4f}" for i in range(3)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Mejorado SVM — Heart Disease ({ITERATIONS} iters)", fontsize=13)
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Mejorado Kernel Quality — Heart Disease", fontsize=13)
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

    print("\n✓ Plots guardados en improved/results/")

    np.save(os.path.join(out, "mean_scores.npy"), {m: ms[m] for m in METHODS})
    np.save(os.path.join(out, "std_scores.npy"),  {m: ss[m] for m in METHODS})

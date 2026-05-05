#!/usr/bin/env python3
"""
Comparación: Baseline vs Mejorado — Heart Disease + Extremality MKL
Corre ambas versiones y muestra la diferencia en accuracy, F1 y AUC.
"""
import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Carga dinámica de los módulos base e improved ─────────────────────────────
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Cargando módulos...")
base_mod = load_module("heart_base",     os.path.join(BASE_DIR, "base",     "heart_base.py"))
impr_mod = load_module("heart_improved", os.path.join(BASE_DIR, "improved", "heart_improved.py"))

# ── Parámetros de comparación ─────────────────────────────────────────────────
ITERATIONS = 30

SCORE_LABELS = ['Accuracy', 'F1', 'AUC']
BASE_COLORS  = ['#90CAF9', '#EF9A9A', '#A5D6A7', '#FFCC80']   # pastel
IMPR_COLORS  = ['#1565C0', '#B71C1C', '#2E7D32', '#E65100']   # sólido

# ── Correr simulaciones ───────────────────────────────────────────────────────
print(f"\nBaseline ({ITERATIONS} iteraciones)...")
_, _, ms_base, ss_base = base_mod.run_simulation(base_mod.X_raw, base_mod.y, ITERATIONS)

print(f"Mejorado ({ITERATIONS} iteraciones)...")
_, _, ms_impr, ss_impr = impr_mod.run_simulation(impr_mod.X_raw, impr_mod.y, ITERATIONS)

BASE_METHODS = list(ms_base.keys())
IMPR_METHODS = list(ms_impr.keys())

# ── Tabla comparativa ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COMPARACIÓN: BASELINE vs MEJORADO")
print("=" * 72)
print(f"{'Versión / Método':<28}" + "".join(f"{n:>14}" for n in SCORE_LABELS))
print("-" * 72)

print("── BASELINE ──")
for m in BASE_METHODS:
    row = f"  {m:<26}" + "".join(f"{ms_base[m][i]:>8.4f} ±{ss_base[m][i]:.3f}" for i in range(3))
    print(row)

print("── MEJORADO ──")
for m in IMPR_METHODS:
    row = f"  {m:<26}" + "".join(f"{ms_impr[m][i]:>8.4f} ±{ss_impr[m][i]:.3f}" for i in range(3))
    print(row)

# Mejor método de cada versión
best_base = max(BASE_METHODS, key=lambda m: ms_base[m][2])  # por AUC
best_impr = max(IMPR_METHODS, key=lambda m: ms_impr[m][2])

print("\n── Mejores métodos (por AUC) ──")
print(f"  Baseline: {best_base:25} — AUC={ms_base[best_base][2]:.4f}, "
      f"Acc={ms_base[best_base][0]:.4f}, F1={ms_base[best_base][1]:.4f}")
print(f"  Mejorado: {best_impr:25} — AUC={ms_impr[best_impr][2]:.4f}, "
      f"Acc={ms_impr[best_impr][0]:.4f}, F1={ms_impr[best_impr][1]:.4f}")

delta_auc = ms_impr[best_impr][2] - ms_base[best_base][2]
delta_acc = ms_impr[best_impr][0] - ms_base[best_base][0]
delta_f1  = ms_impr[best_impr][1] - ms_base[best_base][1]
print(f"\n  Δ AUC = {delta_auc:+.4f}  |  Δ Acc = {delta_acc:+.4f}  |  Δ F1 = {delta_f1:+.4f}")
print("=" * 72)

# ── Plot 1: Barras agrupadas Baseline vs Mejorado ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Comparación: Baseline vs Mejorado — Heart Disease", fontsize=14)

x = np.arange(len(BASE_METHODS))
width = 0.35

for si, (score_label, ax) in enumerate(zip(SCORE_LABELS, axes)):
    b_means = [ms_base[m][si] for m in BASE_METHODS]
    b_stds  = [ss_base[m][si] for m in BASE_METHODS]
    i_means = [ms_impr[m][si] for m in IMPR_METHODS]
    i_stds  = [ss_impr[m][si] for m in IMPR_METHODS]

    bars_b = ax.bar(x - width/2, b_means, width, yerr=b_stds,
                    label='Baseline', color=BASE_COLORS, alpha=0.9,
                    capsize=4, ecolor='gray')
    bars_i = ax.bar(x + width/2, i_means, width, yerr=i_stds,
                    label='Mejorado', color=IMPR_COLORS, alpha=0.9,
                    capsize=4, ecolor='#555')

    ax.set_title(score_label, fontsize=12)
    ax.set_ylim(0.6, 1.05)
    ax.set_xticks(x)
    short_labels = ['Natural\nMKL', 'Anti-Natural\nMKL', 'RBF', 'Poly (d=3)']
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
out = BASE_DIR
plt.savefig(os.path.join(out, "comparison_bars.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nGuardado: comparison_bars.png")

# ── Plot 2: Radar / lineas — mejor de cada versión ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"Mejor método: Baseline ({best_base}) vs Mejorado ({best_impr})",
             fontsize=12)

for si, (score_label, ax) in enumerate(zip(SCORE_LABELS, axes)):
    categories = ['Baseline', 'Mejorado']
    values     = [ms_base[best_base][si], ms_impr[best_impr][si]]
    stds       = [ss_base[best_base][si], ss_impr[best_impr][si]]
    colors     = ['#42A5F5', '#EF5350']

    bars = ax.bar(categories, values, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, ecolor='gray', width=0.45)
    ax.set_title(score_label, fontsize=12)
    ax.set_ylim(0.7, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Flecha de delta
    delta = values[1] - values[0]
    sign  = '+' if delta >= 0 else ''
    color = '#2E7D32' if delta >= 0 else '#B71C1C'
    ax.text(0.5, 0.05, f'Δ = {sign}{delta:.4f}',
            transform=ax.transAxes, ha='center', fontsize=11,
            color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out, "comparison_best.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: comparison_best.png")

print("\n✓ Comparación completa.")

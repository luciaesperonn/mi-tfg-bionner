"""
Figuras estaticas (PNG, estilo matplotlib por defecto: dpi=150, grid alpha=0.3,
igual que el resto del repo) para la memoria del TFG, a partir del barrido fino
multiseed ya calculado:
  outputs/multiseed/multiseed_finegrid_curvas.csv    (curva completa: fine_resweep_curvas.py)
  outputs/multiseed/multiseed_finegrid_resultados.csv (resumen por combo: fine_resweep.py)

Genera:
  1) comparacion_threshold_sweep_fino_multiseed.png -- media +/- std por encoder
  2) threshold_sweep_fino_seeds_por_encoder.png      -- 1 panel por encoder, 3 seeds
  3) f1_macro_optimo_mean_std_por_encoder.png        -- barras, media +/- std
  4) caida_f1_por_encoder_seed.png                   -- fragilidad del optimo
  5) ranking_estabilidad_seeds.png                   -- estabilidad del ranking entre seeds

Uso: python baseline/multiseed_figuras.py  (desde la raiz del repo o desde baseline/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "multiseed"

curves = pd.read_csv(OUT / "multiseed_finegrid_curvas.csv")
summary = pd.read_csv(OUT / "multiseed_finegrid_resultados.csv")

ENCODERS = [
    ("pubmedbert", "PubMedBERT"),
    ("biolinkbert_base", "BioLinkBERT-base"),
    ("biobert", "BioBERT"),
    ("scibert", "SciBERT"),
]
SEEDS = [42, 123, 2024]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # mismo ciclo por defecto que ya usa el repo
SEED_ALPHA = {42: 1.0, 123: 0.65, 2024: 0.4}


# =====================================================================
# 1) Comparacion global: media +/- std por encoder, en todo el grid fino
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 6))
for i, (key, label) in enumerate(ENCODERS):
    g = curves[curves["exp"] == key]
    pivot = g.pivot(index="threshold", columns="seed", values="macro_f1")
    th = pivot.index.values
    mean = pivot.mean(axis=1).values
    std = pivot.std(axis=1).values
    color = COLORS[i % len(COLORS)]
    ax.plot(th, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(th, mean - std, mean + std, color=color, alpha=0.15)
    best_idx = np.argmax(mean)
    ax.scatter([th[best_idx]], [mean[best_idx]], color=color, s=90, zorder=5,
               edgecolor="black", linewidth=1.2)

ax.set_xlabel("threshold (prob. minima para predecir relacion en vez de no_relation)")
ax.set_ylabel("Macro F1 (media entre 3 seeds)")
ax.set_title("Calibracion de umbral en blind (grid fino) -- media +/- std, 3 seeds\n"
             "(circulos con borde negro = mejor threshold de la media)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(OUT / "comparacion_threshold_sweep_fino_multiseed.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: comparacion_threshold_sweep_fino_multiseed.png")


# =====================================================================
# 2) Pequenios multiples: 1 panel por encoder, 3 seeds superpuestas
#    (para ver forma de la curva: meseta vs pico afilado)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.ravel()

for i, (key, label) in enumerate(ENCODERS):
    ax = axes[i]
    color = COLORS[i % len(COLORS)]
    g = curves[curves["exp"] == key]
    for seed in SEEDS:
        s = g[g["seed"] == seed].sort_values("threshold")
        ax.plot(s["threshold"], s["macro_f1"], color=color, alpha=SEED_ALPHA[seed],
                linewidth=1.6, label=f"seed {seed}")
        best_row = s.loc[s["macro_f1"].idxmax()]
        ax.scatter([best_row["threshold"]], [best_row["macro_f1"]], color=color,
                   alpha=SEED_ALPHA[seed], s=55, zorder=5, edgecolor="black", linewidth=0.8)

    row = summary[(summary["exp"] == key)]
    shapes = ", ".join(f"s{r.seed}={r.forma_curva.split(' ')[0].lower()}" for r in row.itertuples())
    ax.set_title(f"{label}\n({shapes})", fontsize=11)
    ax.set_xlabel("threshold")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Barrido fino de threshold por seed (0.85-0.9999) -- forma de la curva cerca del optimo",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "threshold_sweep_fino_seeds_por_encoder.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: threshold_sweep_fino_seeds_por_encoder.png")


# =====================================================================
# 3) Barras: F1 macro optimo (media +/- std) por encoder
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
agg = summary.groupby("exp")["macro_f1_fino"].agg(["mean", "std"])
agg = agg.reindex([k for k, _ in ENCODERS])
labels = [label for _, label in ENCODERS]
means = agg["mean"].values
stds = agg["std"].values

order = np.argsort(-means)
ax.bar(np.arange(len(labels)), means[order], yerr=stds[order], capsize=6,
       color=[COLORS[o % len(COLORS)] for o in order])
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels([labels[o] for o in order])
for x, m in zip(np.arange(len(labels)), means[order]):
    ax.text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=10)

ax.set_ylabel("Macro F1 (optimo del grid fino)")
ax.set_title("F1 macro optimo por encoder -- media +/- std entre 3 seeds")
ax.set_ylim(0, max(means + stds) * 1.2)
ax.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "f1_macro_optimo_mean_std_por_encoder.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: f1_macro_optimo_mean_std_por_encoder.png")


# =====================================================================
# 4) Caida de F1 (F1_optimo - F1_a_th-0.05) por encoder x seed
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 6))
n_enc = len(ENCODERS)
group_w = 0.75
bar_w = group_w / len(SEEDS)

for i, (key, label) in enumerate(ENCODERS):
    color = COLORS[i % len(COLORS)]
    rows = summary[summary["exp"] == key].set_index("seed")
    for j, seed in enumerate(SEEDS):
        x = i + (j - (len(SEEDS) - 1) / 2) * bar_w
        val = rows.loc[seed, "caida_f1"]
        ax.bar(x, val, width=bar_w * 0.92, color=color, alpha=SEED_ALPHA[seed],
               edgecolor="black", linewidth=0.4,
               label=f"seed {seed}" if i == 0 else None)

ax.axhline(0.02, color="gray", linestyle="--", linewidth=1, label="umbral meseta (<0.02)")
ax.axhline(0.05, color="black", linestyle=":", linewidth=1, label="umbral pico afilado (>0.05)")

ax.set_xticks(np.arange(n_enc))
ax.set_xticklabels([label for _, label in ENCODERS])
ax.set_ylabel("Caida de F1 (F1 optimo - F1 a threshold-0.05)")
ax.set_title("Fragilidad del threshold optimo -- caida de F1 por encoder y seed")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "caida_f1_por_encoder_seed.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: caida_f1_por_encoder_seed.png")


# =====================================================================
# 5) Estabilidad del ranking: F1 optimo por seed, una linea por encoder
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(SEEDS))

for i, (key, label) in enumerate(ENCODERS):
    color = COLORS[i % len(COLORS)]
    rows = summary[summary["exp"] == key].set_index("seed").loc[SEEDS]
    ax.plot(x, rows["macro_f1_fino"].values, marker="o", markersize=8,
            color=color, linewidth=2, label=label)

ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_ylabel("Macro F1 (optimo del grid fino)")
ax.set_title("Estabilidad del ranking entre encoders segun la seed")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(OUT / "ranking_estabilidad_seeds.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: ranking_estabilidad_seeds.png")

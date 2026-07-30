"""
Figuras de precision/recall a partir del barrido con metricas extra
(outputs/multiseed/multiseed_finegrid_pr.csv de fine_resweep_pr.py, mas
outputs/multiseed/multiseed_coarse_pr.csv de fine_resweep_pr_coarse.py).
Mismo estilo que multiseed_figuras.py: matplotlib por defecto, dpi=150, grid
alpha=0.3.

Genera:
  1) precision_recall_f1_vs_threshold.png -- pequenios multiples, 1 panel/encoder,
     P/R/F1 macro (tipado, mismo promedio que macro_f1) vs threshold, media entre
     seeds. Usa SOLO el grid fino (0.85-0.9999): es la zona de calibracion cerca
     del optimo, no hace falta el rango completo para esta.
  2) curva_pr.png  -- precision vs recall "binario" (existe relacion o no),
     una curva por encoder, media entre seeds. Usa el rango COMPLETO
     (0.001-0.9999) para que el AUC-PR sea el area real bajo toda la curva.
  3) curva_roc.png -- TPR vs FPR "binario", una curva por encoder, media entre
     seeds, con diagonal de referencia. Tambien rango completo.

Uso: python baseline/multiseed_figuras_pr.py  (desde la raiz del repo o desde baseline/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "multiseed"
DATA_DIR = REPO / "data" / "english"

df = pd.read_csv(OUT / "multiseed_finegrid_pr.csv")
df_full = pd.concat(
    [pd.read_csv(OUT / "multiseed_coarse_pr.csv"), df],
    ignore_index=True,
).sort_values(["exp", "seed", "threshold"])

# Extremos triviales de la curva (no hace falta calcularlos: a threshold=0 se
# predice TODO el pool -> FPR=1,TPR=1 exactos; a threshold=1 no se predice
# nada -> FPR=0,TPR=0 exactos). Se usan para cerrar PR/ROC hasta el rango
# completo [0,1] sin tener que barrer el tramo carisimo cerca de threshold=0.
gold_dev = pd.read_csv(DATA_DIR / "eng-dev-rel.tsv", sep="\t")
TOTAL_CANDIDATES = sum(1 for l in open(DATA_DIR / "eng_dev_blind.txt", encoding="utf-8") if l.strip())
TOTAL_POSITIVES = gold_dev.drop_duplicates(subset=["document_id", "head_span", "tail_span"]).shape[0]
PREVALENCE = TOTAL_POSITIVES / TOTAL_CANDIDATES  # precision en el extremo "predecir todo"

ENCODERS = [
    ("pubmedbert", "PubMedBERT"),
    ("biolinkbert_base", "BioLinkBERT-base"),
    ("biobert", "BioBERT"),
    ("scibert", "SciBERT"),
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def auc_trapz(x, y):
    """AUC por la regla del trapecio, ordenando por x ascendente."""
    order = np.argsort(x)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(np.array(y)[order], np.array(x)[order]))


# =====================================================================
# 1) P / R / F1 macro (tipado) vs threshold -- pequenios multiples
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.ravel()

for i, (key, label) in enumerate(ENCODERS):
    ax = axes[i]
    color = COLORS[i % len(COLORS)]
    g = df[df["exp"] == key].groupby("threshold")[["macro_f1", "macro_precision", "macro_recall"]].mean()
    th = g.index.values

    ax.plot(th, g["macro_f1"], color=color, linewidth=2.2, label="F1 macro")
    ax.plot(th, g["macro_precision"], color=color, linewidth=1.6, linestyle="--", alpha=0.75, label="Precision macro")
    ax.plot(th, g["macro_recall"], color=color, linewidth=1.6, linestyle=":", alpha=0.75, label="Recall macro")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Precision / Recall / F1 macro vs threshold (grid fino, media entre 3 seeds)\n"
             "-- explica por que el F1 cae cerca del borde: el recall se hunde", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "precision_recall_f1_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: precision_recall_f1_vs_threshold.png")


# =====================================================================
# 2) Curva PR (precision vs recall, binario: relacion vs no_relation)
#    rango completo de threshold (0.001-0.9999) para que el AUC sea real
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 6.5))
for i, (key, label) in enumerate(ENCODERS):
    color = COLORS[i % len(COLORS)]
    g = df_full[df_full["exp"] == key].groupby("threshold")[["precision_bin", "recall_bin"]].mean()
    # cierre con los extremos triviales: threshold=1 -> (recall=0, precision=1
    # por convencion, igual que sklearn); threshold=0 -> (recall=1, precision=prevalencia)
    recall = np.concatenate([[0.0], g["recall_bin"].values, [1.0]])
    precision = np.concatenate([[1.0], g["precision_bin"].values, [PREVALENCE]])
    aucpr = auc_trapz(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f"{label} (AUC-PR={aucpr:.3f})")

ax.set_xlabel("Recall (binario: relacion detectada vs no)")
ax.set_ylabel("Precision (binario)")
ax.set_title("Curva PR -- deteccion de relacion vs no_relation\n"
             "(threshold 0.001-0.9999 + extremos triviales 0/1, media entre 3 seeds)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(OUT / "curva_pr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: curva_pr.png")


# =====================================================================
# 3) Curva ROC (TPR vs FPR, binario) -- rango completo de threshold
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 6.5))
for i, (key, label) in enumerate(ENCODERS):
    color = COLORS[i % len(COLORS)]
    g = df_full[df_full["exp"] == key].groupby("threshold")[["fpr_bin", "recall_bin"]].mean()
    # cierre con los extremos triviales: threshold=1 -> (0,0); threshold=0 -> (1,1) exactos
    fpr = np.concatenate([[0.0], g["fpr_bin"].values, [1.0]])
    tpr = np.concatenate([[0.0], g["recall_bin"].values, [1.0]])
    aucroc = auc_trapz(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{label} (AUC-ROC={aucroc:.3f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="azar")
ax.set_xlabel("False Positive Rate (binario)")
ax.set_ylabel("True Positive Rate / Recall (binario)")
ax.set_title("Curva ROC -- deteccion de relacion vs no_relation\n"
             "(threshold 0.001-0.9999 + extremos triviales 0/1, media entre 3 seeds)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(OUT / "curva_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: curva_roc.png")

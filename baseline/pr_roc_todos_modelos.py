"""
Curvas PR / ROC / precision-recall-F1-vs-threshold para TODOS los modelos
(los 4 encoders base + XLM-RoBERTa + DeBERTa-v3 + XLNet + T5), mismo estilo y
misma definicion de metricas que multiseed_figuras_pr.py / fine_resweep_pr.py:

  - "binario" = existe relacion vs no_relation (ignora el tipo exacto),
    para las curvas PR y ROC.
  - macro_precision/macro_recall = misma media "por-relacion" que macro_f1.

Los 4 encoders base reusan el barrido multiseed YA calculado (3 seeds reales,
outputs/multiseed/multiseed_finegrid_pr.csv + multiseed_coarse_pr.csv) -- no
se recalcula nada para ellos. Para XLM-RoBERTa (solo seed42), DeBERTa-v3,
XLNet y T5 se recalcula sobre un grid combinado (grueso 0.001-0.95 + fino
0.95-0.999) reusando blind_probs_<exp>.npy ya guardado -- no hace falta
reentrenar ni volver a inferir sobre el pool blind.

Genera en outputs/comparacion_todos_modelos/:
  1) curva_pr_todos_modelos.png
  2) curva_roc_todos_modelos.png
  3) precision_recall_f1_vs_threshold_nuevos.png (xlm_roberta/deberta_v3/xlnet/t5)
  4) pr_roc_todos_modelos.csv (metricas crudas, por si hace falta reusarlas)

Uso: python baseline/pr_roc_todos_modelos.py (desde la raiz del repo o baseline/)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baseline"))
from score import evaluate  # noqa: E402

DATA_DIR = REPO / "data" / "english"
OUTPUTS = REPO / "outputs"
OUT = OUTPUTS / "comparacion_todos_modelos"
OUT.mkdir(exist_ok=True)

with open(DATA_DIR / "rel2id.json") as f:
    rel2id = json.load(f)
id2rel = {v: k for k, v in rel2id.items()}
NO_REL_ID = rel2id["no_relation"]

BLIND_DEV_PATH = DATA_DIR / "eng_dev_blind.txt"
BLIND_GOLD_TSV = DATA_DIR / "eng-dev-rel.tsv"
blind_raw = [json.loads(l) for l in open(BLIND_DEV_PATH, encoding="utf-8") if l.strip()]
gold_df_blind = pd.read_csv(BLIND_GOLD_TSV, sep="\t")
TOTAL_CANDIDATES = len(blind_raw)


def instance_key(doc_id, head_span, tail_span):
    return f"{doc_id}|{head_span}|{tail_span}"


GOLD_KEYS = {
    instance_key(str(r.document_id), str(r.head_span), str(r.tail_span))
    for r in gold_df_blind.itertuples()
}
TOTAL_POSITIVES = len(GOLD_KEYS)
PREVALENCE = TOTAL_POSITIVES / TOTAL_CANDIDATES

MODELS_NEW = [
    ("1H-xlm-roberta/seed42", "xlm_roberta", "XLM-RoBERTa"),
    ("1I-deberta-v3", "deberta_v3", "DeBERTa-v3"),
    ("1J-xlnet", "xlnet", "XLNet"),
    ("1K-t5-scifive", "t5", "T5 (SciFive)"),
]
MODELS_MULTISEED = [
    ("pubmedbert", "PubMedBERT"),
    ("biolinkbert_base", "BioLinkBERT-base"),
    ("biobert", "BioBERT"),
    ("scibert", "SciBERT"),
]
ALL_LABELS = [l for _, l in MODELS_MULTISEED] + [l for _, _, l in MODELS_NEW]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_OF = {l: COLORS[i % len(COLORS)] for i, l in enumerate(ALL_LABELS)}

# grid combinado: grueso hasta 0.95, fino de 0.95 a 0.999 (zona de calibracion)
GRID = sorted(set(
    [0.001, 0.005] + [round(float(x), 3) for x in np.arange(0.01, 0.951, 0.02)] +
    [round(float(x), 4) for x in np.arange(0.95, 0.9901, 0.005)] +
    [round(float(x), 4) for x in np.arange(0.990, 0.9991, 0.002)] + [0.9995, 0.9999]
))


def preds_at_threshold(probs, threshold):
    probs2 = probs.copy()
    probs2[:, NO_REL_ID] = -1
    best_rel_id = probs2.argmax(axis=1)
    best_rel_prob = probs[np.arange(len(probs)), best_rel_id]
    return np.where(best_rel_prob >= threshold, best_rel_id, NO_REL_ID)


def rows_from_preds(pred_ids):
    labels = [id2rel[i] for i in pred_ids]
    return [
        {"document_id": inst["doc_id"], "relation": rel,
         "head_text": inst["h"]["name"], "head_span": inst["head_span"], "head_type": inst["head_type"],
         "tail_text": inst["t"]["name"], "tail_span": inst["tail_span"], "tail_type": inst["tail_type"]}
        for inst, rel in zip(blind_raw, labels) if rel != "no_relation"
    ]


def metrics_at(probs, threshold):
    pred_ids = preds_at_threshold(probs, threshold)
    pred_rows = rows_from_preds(pred_ids)
    res = evaluate(pd.DataFrame(pred_rows), gold_df_blind)
    per_rel = res["per_relation"]
    macro_precision = np.mean([v["precision"] for v in per_rel.values()]) if per_rel else 0.0
    macro_recall = np.mean([v["recall"] for v in per_rel.values()]) if per_rel else 0.0

    pred_keys = {instance_key(r["document_id"], r["head_span"], r["tail_span"]) for r in pred_rows}
    tp_bin = len(GOLD_KEYS & pred_keys)
    fp_bin = len(pred_keys - GOLD_KEYS)
    fn_bin = len(GOLD_KEYS - pred_keys)
    tn_bin = TOTAL_CANDIDATES - len(GOLD_KEYS | pred_keys)

    precision_bin = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0.0
    recall_bin = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
    fpr_bin = fp_bin / (fp_bin + tn_bin) if (fp_bin + tn_bin) > 0 else 0.0
    return {"macro_f1": res["macro_f1"], "macro_precision": macro_precision, "macro_recall": macro_recall,
            "precision_bin": precision_bin, "recall_bin": recall_bin, "fpr_bin": fpr_bin}


def auc_trapz(x, y):
    order = np.argsort(x)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(np.array(y)[order], np.array(x)[order]))


# =====================================================================
# 1) Metricas: multiseed (reusadas) + nuevas (recalculadas sobre grid)
# =====================================================================
MS_OUT = OUTPUTS / "multiseed"
ms_fine = pd.read_csv(MS_OUT / "multiseed_finegrid_pr.csv")
ms_coarse = pd.read_csv(MS_OUT / "multiseed_coarse_pr.csv")
ms_full = pd.concat([ms_coarse, ms_fine], ignore_index=True)

rows_new = []
for outdir, exp, label in MODELS_NEW:
    npy_path = OUTPUTS / outdir / f"blind_probs_{exp}.npy"
    if not npy_path.exists():
        print(f"[AVISO] falta {npy_path}, se omite {label}")
        continue
    print(f"[pr/roc] {label} -- {len(GRID)} thresholds sobre probs cacheadas...", flush=True)
    probs = np.load(npy_path)
    for th in GRID:
        m = metrics_at(probs, th)
        rows_new.append({"exp": exp, "label": label, "threshold": th, **m})
    print(f"  listo {label}", flush=True)

df_new = pd.DataFrame(rows_new)
df_new.to_csv(OUT / "pr_roc_nuevos_modelos.csv", index=False)
print("guardado: pr_roc_nuevos_modelos.csv")


# =====================================================================
# 2) Curva PR -- los 8 modelos
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for exp, label in MODELS_MULTISEED:
    color = COLOR_OF[label]
    g = ms_full[ms_full["exp"] == exp].groupby("threshold")[["precision_bin", "recall_bin"]].mean()
    recall = np.concatenate([[1.0], g["recall_bin"].values, [0.0]])
    precision = np.concatenate([[PREVALENCE], g["precision_bin"].values, [1.0]])
    aucpr = auc_trapz(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2, label=f"{label} (AUC-PR={aucpr:.3f}, 3 seeds)")

for outdir, exp, label in MODELS_NEW:
    g = df_new[df_new["exp"] == exp].sort_values("threshold")
    if g.empty:
        continue
    color = COLOR_OF[label]
    recall = np.concatenate([[1.0], g["recall_bin"].values, [0.0]])
    precision = np.concatenate([[PREVALENCE], g["precision_bin"].values, [1.0]])
    aucpr = auc_trapz(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2, label=f"{label} (AUC-PR={aucpr:.3f})")

ax.set_xlabel("Recall (binario: relacion detectada vs no)")
ax.set_ylabel("Precision (binario)")
ax.set_title("Curva PR -- deteccion de relacion vs no_relation, los 8 modelos\n"
             "(4 base = media de 3 seeds; XLM-RoBERTa/DeBERTa-v3/XLNet/T5 = 1 run)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.savefig(OUT / "curva_pr_todos_modelos.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: curva_pr_todos_modelos.png")


# =====================================================================
# 3) Curva ROC -- los 8 modelos
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for exp, label in MODELS_MULTISEED:
    color = COLOR_OF[label]
    g = ms_full[ms_full["exp"] == exp].groupby("threshold")[["fpr_bin", "recall_bin"]].mean()
    fpr = np.concatenate([[1.0], g["fpr_bin"].values, [0.0]])
    tpr = np.concatenate([[1.0], g["recall_bin"].values, [0.0]])
    aucroc = auc_trapz(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC-ROC={aucroc:.3f}, 3 seeds)")

for outdir, exp, label in MODELS_NEW:
    g = df_new[df_new["exp"] == exp].sort_values("threshold")
    if g.empty:
        continue
    color = COLOR_OF[label]
    fpr = np.concatenate([[1.0], g["fpr_bin"].values, [0.0]])
    tpr = np.concatenate([[1.0], g["recall_bin"].values, [0.0]])
    aucroc = auc_trapz(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC-ROC={aucroc:.3f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="azar")
ax.set_xlabel("False Positive Rate (binario)")
ax.set_ylabel("True Positive Rate / Recall (binario)")
ax.set_title("Curva ROC -- deteccion de relacion vs no_relation, los 8 modelos")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.savefig(OUT / "curva_roc_todos_modelos.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: curva_roc_todos_modelos.png")


# =====================================================================
# 4) Precision / Recall / F1 macro (tipado) vs threshold -- solo los 4
#    modelos nuevos (los 4 base ya tienen su version en
#    precision_recall_f1_vs_threshold.png)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.ravel()
for i, (outdir, exp, label) in enumerate(MODELS_NEW):
    ax = axes[i]
    g = df_new[df_new["exp"] == exp].sort_values("threshold")
    if g.empty:
        ax.set_title(f"{label} (sin datos)")
        continue
    color = COLOR_OF[label]
    ax.plot(g["threshold"], g["macro_f1"], color=color, linewidth=2.2, label="F1 macro")
    ax.plot(g["threshold"], g["macro_precision"], color=color, linewidth=1.6, linestyle="--",
            alpha=0.75, label="Precision macro")
    ax.plot(g["threshold"], g["macro_recall"], color=color, linewidth=1.6, linestyle=":",
            alpha=0.75, label="Recall macro")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Precision / Recall / F1 macro vs threshold -- XLM-RoBERTa, DeBERTa-v3, XLNet, T5", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "precision_recall_f1_vs_threshold_nuevos.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: precision_recall_f1_vs_threshold_nuevos.png")

print("\nTODAS LAS CURVAS PR/ROC GENERADAS")

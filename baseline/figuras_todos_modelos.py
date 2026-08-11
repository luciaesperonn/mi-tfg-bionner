"""
Figuras comparativas (PNG, dpi=150, grid alpha=0.3 -- mismo estilo que
multiseed_figuras.py) de TODOS los modelos con entrenamiento completo hasta
la fecha: los 4 encoders base, XLM-RoBERTa, DeBERTa-v3, XLNet y T5 (SciFive).

Quedan fuera 1E-biomedbert-large y 1F-biolinkbert-large (nunca terminaron:
solo tienen checkpoint, sin history/results/threshold_sweep).

Fuentes:
  outputs/comparacion_encoders.csv        -- tabla maestra (dev curado + ciego)
  outputs/multiseed_resultados.csv        -- calibracion con 3 seeds reales
                                              (pubmedbert/biolinkbert_base/
                                              biobert/scibert/xlm_roberta)
  outputs/<outdir>/history_<exp>.json     -- curvas de entrenamiento
  outputs/<outdir>/threshold_sweep_<exp>.csv (o seed42/ para xlm_roberta)

Genera en outputs/comparacion_todos_modelos/:
  1) bar_macro_f1_dev.png           -- macro F1 dev (curado), los 8 modelos
  2) bar_macro_f1_ciego.png         -- argmax vs calibrado, con std donde hay
                                        multiseed y aviso de borde de grid
  3) curvas_entrenamiento.png       -- val_macro_f1 y train_loss por epoch
  4) threshold_sweep_comparativo.png -- macro_f1 vs threshold, los 8 modelos

Uso: python baseline/figuras_todos_modelos.py (desde la raiz del repo o baseline/)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUT = OUTPUTS / "comparacion_todos_modelos"
OUT.mkdir(exist_ok=True)

# (outdir, exp, label, subdir_para_history_y_threshold)
MODELS = [
    ("1A-pubmedbert",  "pubmedbert",        "PubMedBERT",        ""),
    ("1B-biolinkbert", "biolinkbert_base",  "BioLinkBERT-base",  ""),
    ("1C-biobert",     "biobert",           "BioBERT",           ""),
    ("1D-scibert",     "scibert",           "SciBERT",           ""),
    ("1H-xlm-roberta", "xlm_roberta",       "XLM-RoBERTa",       "seed42/"),
    ("1I-deberta-v3",  "deberta_v3",        "DeBERTa-v3",        ""),
    ("1J-xlnet",       "xlnet",             "XLNet",             ""),
    ("1K-t5-scifive",  "t5",                "T5 (SciFive)",      ""),
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_OF = {label: COLORS[i % len(COLORS)] for i, (_, _, label, _) in enumerate(MODELS)}

# =====================================================================
# Carga de datos
# =====================================================================
tab = pd.read_csv(OUTPUTS / "comparacion_encoders.csv")
ms = pd.read_csv(OUTPUTS / "multiseed_resultados.csv")

MULTISEED_EXP = {"pubmedbert", "biolinkbert_base", "biobert", "scibert", "xlm_roberta"}

LABEL_TO_ROW = {}
for outdir, exp, label, sub in MODELS:
    row = {"exp": exp, "label": label, "outdir": outdir, "sub": sub}
    if exp in MULTISEED_EXP:
        sub_ms = ms[ms["exp"] == exp]
        row["macro_f1_ciego_calibrado_mean"] = sub_ms["macro_f1_ciego_calibrado"].mean()
        row["macro_f1_ciego_calibrado_std"] = sub_ms["macro_f1_ciego_calibrado"].std()
        row["borde_grid"] = bool(sub_ms["threshold_en_borde_del_grid"].any())
        row["macro_f1_ciego_argmax"] = sub_ms["macro_f1_ciego_argmax"].mean()
        row["n_seeds"] = len(sub_ms)
    # Match robusto: los labels de comparacion_encoders.csv empiezan todos
    # por el codigo del outdir ("1A", "1I", ...) -- un str.contains(exp)
    # falla en biolinkbert_base/deberta_v3 porque el guion bajo del exp no
    # coincide con el guion del label ("BioLinkBERT-base", "1I-deberta-v3").
    tr = tab[tab["exp"].str.startswith(outdir[:2])]
    if exp not in MULTISEED_EXP:
        if not tr.empty:
            r = tr.iloc[0]
            row["macro_f1_ciego_calibrado_mean"] = r.get("macro_f1_ciego_calibrado", np.nan)
            row["macro_f1_ciego_calibrado_std"] = 0.0
            row["borde_grid"] = False
            row["macro_f1_ciego_argmax"] = r.get("macro_f1_ciego", np.nan)
            row["n_seeds"] = 1
        else:
            row["macro_f1_ciego_calibrado_mean"] = np.nan
            row["macro_f1_ciego_calibrado_std"] = 0.0
            row["borde_grid"] = False
            row["macro_f1_ciego_argmax"] = np.nan
            row["n_seeds"] = 1
    if not tr.empty:
        row["macro_f1_dev"] = tr.iloc[0]["macro_f1"]
    elif exp in MULTISEED_EXP:
        # xlm_roberta no tiene fila en comparacion_encoders.csv (nunca se
        # corrio en el formato "curado" de una sola seed) -- se usa la
        # media del "macro_f1_curado" multiseed como sustituto.
        row["macro_f1_dev"] = ms[ms["exp"] == exp]["macro_f1_curado"].mean()
    else:
        row["macro_f1_dev"] = np.nan
    LABEL_TO_ROW[label] = row

# threshold_sweep_xlnet.csv fue calibrado luego (run_1i1j_blind_calibrated.py) --
# el "borde_grid" de xlnet no viene de comparacion_encoders.csv (que no lo guarda),
# se lee directo del results_blind_xlnet.json.
for outdir, exp, label, sub in MODELS:
    if exp in ("deberta_v3", "xlnet", "t5"):
        bp = OUTPUTS / outdir / f"results_blind_{exp}.json"
        if bp.exists():
            bd = json.load(open(bp))
            LABEL_TO_ROW[label]["borde_grid"] = bool(bd.get("threshold_en_borde_del_grid", False))
            LABEL_TO_ROW[label]["best_threshold"] = bd.get("best_threshold")

BASELINE = 0.6944  # mismo baseline de referencia que comparacion_encoders.csv


# =====================================================================
# 1) Macro F1 dev (curado) -- los 8 modelos
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 6))
order = sorted(LABEL_TO_ROW, key=lambda l: -LABEL_TO_ROW[l]["macro_f1_dev"])
vals = [LABEL_TO_ROW[l]["macro_f1_dev"] for l in order]
colors = [COLOR_OF[l] for l in order]
ax.bar(np.arange(len(order)), vals, color=colors)
ax.axhline(BASELINE, color="gray", linestyle="--", linewidth=1, label=f"baseline ({BASELINE:.4f})")
for x, v in zip(np.arange(len(order)), vals):
    ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Macro F1 (dev curado)")
ax.set_title("Macro F1 en dev curado -- comparacion de los 8 modelos entrenados")
ax.set_ylim(0, max(vals) * 1.15)
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "bar_macro_f1_dev.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: bar_macro_f1_dev.png")


# =====================================================================
# 2) Macro F1 ciego: argmax vs calibrado (+ std donde hay multiseed,
#    aviso de borde de grid)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
order = sorted(LABEL_TO_ROW, key=lambda l: -(LABEL_TO_ROW[l]["macro_f1_ciego_calibrado_mean"] or 0))
x = np.arange(len(order))
w = 0.38

argmax_vals = [LABEL_TO_ROW[l]["macro_f1_ciego_argmax"] for l in order]
cal_vals = [LABEL_TO_ROW[l]["macro_f1_ciego_calibrado_mean"] for l in order]
cal_stds = [LABEL_TO_ROW[l]["macro_f1_ciego_calibrado_std"] for l in order]

ax.bar(x - w / 2, argmax_vals, width=w, color="lightgray", edgecolor="black",
       linewidth=0.5, label="ciego argmax (sin calibrar)")
bars_cal = ax.bar(x + w / 2, cal_vals, width=w, yerr=cal_stds, capsize=4,
                   color=[COLOR_OF[l] for l in order], edgecolor="black", linewidth=0.5,
                   label="ciego calibrado (mejor threshold)")

for xi, l in zip(x, order):
    row = LABEL_TO_ROW[l]
    n = row.get("n_seeds", 1)
    txt = f"{row['macro_f1_ciego_calibrado_mean']:.3f}"
    if n > 1:
        txt += f"\n(media {n} seeds)"
    if row.get("borde_grid"):
        txt += "\n⚠ borde grid"
    ax.text(xi + w / 2, row["macro_f1_ciego_calibrado_mean"] + (row["macro_f1_ciego_calibrado_std"] or 0) + 0.012,
            txt, ha="center", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Macro F1 (blind / ciego)")
ax.set_title("Macro F1 ciego: argmax vs calibrado por threshold\n"
             "(⚠ borde grid = el barrido no encontro techo, umbral optimo real probablemente mayor)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.savefig(OUT / "bar_macro_f1_ciego.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: bar_macro_f1_ciego.png")


# =====================================================================
# 3) Curvas de entrenamiento: val_macro_f1 y train_loss por epoch
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

for outdir, exp, label, sub in MODELS:
    hpath = OUTPUTS / outdir / sub / f"history_{exp}.json"
    if not hpath.exists():
        continue
    hist = json.load(open(hpath))
    epochs = [h["epoch"] for h in hist]
    color = COLOR_OF[label]
    ax1.plot(epochs, [h["val_macro_f1"] for h in hist], marker="o", markersize=3,
              color=color, linewidth=1.6, label=label)
    ax2.plot(epochs, [h["train_loss"] for h in hist], marker="o", markersize=3,
              color=color, linewidth=1.6, label=label)

ax1.set_ylabel("Macro F1 (val, dev curado)")
ax1.set_title("Curvas de entrenamiento -- los 8 modelos (15 epochs)")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.3)

ax2.set_yscale("log")
ax2.set_xlabel("epoch")
ax2.set_ylabel("train loss (escala log)")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: curvas_entrenamiento.png")


# =====================================================================
# 4) Barrido de threshold comparativo (grid grueso, todos los modelos)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6.5))

for outdir, exp, label, sub in MODELS:
    spath = OUTPUTS / outdir / sub / f"threshold_sweep_{exp}.csv"
    if not spath.exists():
        continue
    sweep = pd.read_csv(spath).sort_values("threshold")
    color = COLOR_OF[label]
    ax.plot(sweep["threshold"], sweep["macro_f1"], color=color, linewidth=1.8, label=label)
    best_idx = sweep["macro_f1"].idxmax()
    ax.scatter([sweep.loc[best_idx, "threshold"]], [sweep.loc[best_idx, "macro_f1"]],
               color=color, s=70, zorder=5, edgecolor="black", linewidth=1.0)

ax.set_xlabel("threshold (prob. minima para predecir relacion en vez de no_relation)")
ax.set_ylabel("Macro F1 (ciego)")
ax.set_title("Barrido de threshold en blind -- los 8 modelos\n"
             "(circulos con borde negro = mejor threshold; XLM-RoBERTa es solo seed 42)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.savefig(OUT / "threshold_sweep_comparativo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("guardado: threshold_sweep_comparativo.png")

print("\nTodas las figuras guardadas en:", OUT)

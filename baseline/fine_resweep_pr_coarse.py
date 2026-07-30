"""
Complementa fine_resweep_pr.py con el tramo BAJO de threshold (0.001-0.84) que
faltaba para poder trazar curvas PR/ROC completas y con un AUC interpretable
en su rango habitual [0,1] -- el grid fino (0.85-0.9999) solo cubre la zona de
calibracion cerca del optimo, no sirve por si solo para eso.

th=0.0 exacto se evita a proposito: con este modelo casi todos los candidatos
(282k) superan una probabilidad de practicamente cero, lo que dispara el coste
de evaluate() (~10s esa unica llamada vs <1s para el resto). th=0.001 ya cae a
~17k predicciones, coste normal.

Uso: python baseline/fine_resweep_pr_coarse.py  (desde la raiz del repo o desde baseline/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baseline"))
from score import evaluate  # noqa: E402

DATA_DIR = REPO / "data" / "english"
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

MULTISEED_CONFIGS = [
    {"exp": "pubmedbert", "base_outdir": "1A-pubmedbert"},
    {"exp": "biolinkbert_base", "base_outdir": "1B-biolinkbert"},
    {"exp": "biobert", "base_outdir": "1C-biobert"},
    {"exp": "scibert", "base_outdir": "1D-scibert"},
]
SEEDS = [42, 123, 2024]

# 0.85 en adelante ya esta cubierto por multiseed_finegrid_pr.csv
COARSE_GRID = [round(float(x), 4) for x in np.arange(0.001, 0.001 + 0.01 * 84, 0.01)]
COARSE_GRID = sorted(set(th for th in COARSE_GRID if th < 0.85))


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

    return {
        "macro_f1": res["macro_f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "precision_bin": precision_bin,
        "recall_bin": recall_bin,
        "fpr_bin": fpr_bin,
        "tp_bin": tp_bin, "fp_bin": fp_bin, "fn_bin": fn_bin, "tn_bin": tn_bin,
    }


def main():
    rows = []
    for cfg in MULTISEED_CONFIGS:
        for seed in SEEDS:
            out_dir = REPO / "outputs" / cfg["base_outdir"] / f"seed{seed}"
            npy_path = out_dir / f"blind_probs_{cfg['exp']}.npy"
            if not npy_path.exists():
                print(f"falta {npy_path}", file=sys.stderr)
                continue
            print(f"[pr-coarse] {cfg['exp']} seed={seed} ...", flush=True)
            probs = np.load(npy_path)
            for th in COARSE_GRID:
                m = metrics_at(probs, th)
                rows.append({"exp": cfg["exp"], "seed": seed, "threshold": th, **m})

    df = pd.DataFrame(rows)
    out_dir = REPO / "outputs" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiseed_coarse_pr.csv"
    df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}  ({len(df)} filas)")


if __name__ == "__main__":
    main()

"""
Igual que fine_resweep.py pero vuelca la curva COMPLETA (152 puntos de F1 vs
threshold por cada combinacion encoder/seed), no solo el resumen (mejor punto +
forma de la curva). Sirve para graficar las curvas reales en vez de solo el
resumen de 2 puntos.

Uso: python baseline/fine_resweep_curvas.py  (desde la raiz del repo o desde baseline/)
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

MULTISEED_CONFIGS = [
    {"exp": "pubmedbert", "base_outdir": "1A-pubmedbert"},
    {"exp": "biolinkbert_base", "base_outdir": "1B-biolinkbert"},
    {"exp": "biobert", "base_outdir": "1C-biobert"},
    {"exp": "scibert", "base_outdir": "1D-scibert"},
]
SEEDS = [42, 123, 2024]

FINE_GRID = [round(float(x), 4) for x in np.arange(0.85, 0.9901, 0.001)] + \
            [round(float(x), 4) for x in np.arange(0.990, 0.9991, 0.001)] + \
            [0.9995, 0.9999]
FINE_GRID = sorted(set(FINE_GRID))


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


def f1_at(probs, threshold):
    pred_ids = preds_at_threshold(probs, threshold)
    res = evaluate(pd.DataFrame(rows_from_preds(pred_ids)), gold_df_blind)
    return res["macro_f1"]


def main():
    rows = []
    for cfg in MULTISEED_CONFIGS:
        for seed in SEEDS:
            out_dir = REPO / "outputs" / cfg["base_outdir"] / f"seed{seed}"
            npy_path = out_dir / f"blind_probs_{cfg['exp']}.npy"
            if not npy_path.exists():
                print(f"falta {npy_path}", file=sys.stderr)
                continue
            print(f"[curva] {cfg['exp']} seed={seed} ...", flush=True)
            probs = np.load(npy_path)
            for th in FINE_GRID:
                f1 = f1_at(probs, th)
                rows.append({"exp": cfg["exp"], "seed": seed, "threshold": th, "macro_f1": f1})

    df = pd.DataFrame(rows)
    out_dir = REPO / "outputs" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiseed_finegrid_curvas.csv"
    df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}  ({len(df)} filas)")


if __name__ == "__main__":
    main()

"""
Re-barrido fino de threshold sobre las probabilidades YA CACHEADAS
(blind_probs_{exp}.npy) de cada (encoder, seed) -- no reentrena nada, no toca
la GPU, seguro para correr en paralelo con el job de entrenamiento en tmux.

Hace las 4 cosas que hacen falta antes de escribir una conclusion:
 1) Optimo real por combo (grid fino 0.85-0.9999, no solo hasta 0.99)
 2) Media +/- std por encoder (sobre los seeds ya disponibles)
 3) Estabilidad de ranking seed a seed
 4) Forma de la curva: meseta (robusto) vs pico afilado (fragil), comparando
    el F1 en el optimo contra el F1 a threshold-0.05

Uso: python baseline/fine_resweep.py  (desde la raiz del repo o desde baseline/)
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

# grid fino: la duda es si el optimo real supera 0.99 (5/6 corridas ya lo pegaban al borde)
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


def eval_at(probs, threshold):
    pred_ids = preds_at_threshold(probs, threshold)
    return evaluate(pd.DataFrame(rows_from_preds(pred_ids)), gold_df_blind)


def analyze_one(exp, seed, npy_path):
    probs = np.load(npy_path)
    sweep = [(th, eval_at(probs, th)["macro_f1"]) for th in FINE_GRID]
    best_th, best_f1 = max(sweep, key=lambda x: x[1])
    at_edge = best_th == FINE_GRID[-1]

    # forma de la curva: F1 a 0.05 por debajo del optimo (el punto del grid mas cercano)
    ref_th = max(0.85, best_th - 0.05)
    ref_pair = min(sweep, key=lambda x: abs(x[0] - ref_th))
    drop = best_f1 - ref_pair[1]
    shape = "MESETA (robusto)" if drop < 0.02 else ("PICO AFILADO (fragil)" if drop > 0.05 else "intermedio")

    # desglose por relacion EN EL THRESHOLD OPTIMO (no en argmax) -- antes se
    # descartaba, solo se guardaba el macro_f1 agregado
    per_relation = eval_at(probs, best_th)["per_relation"]

    return {
        "exp": exp, "seed": seed,
        "best_threshold_fino": best_th,
        "macro_f1_fino": best_f1,
        "en_borde_del_grid_fino": at_edge,
        "f1_a_th_menos_0.05": ref_pair[1],
        "caida_f1": round(drop, 4),
        "forma_curva": shape,
        "n_puntos_grid": len(FINE_GRID),
        "per_relation": per_relation,
    }


def main():
    rows = []
    per_rel_rows = []
    for cfg in MULTISEED_CONFIGS:
        for seed in SEEDS:
            out_dir = REPO / "outputs" / cfg["base_outdir"] / f"seed{seed}"
            npy_path = out_dir / f"blind_probs_{cfg['exp']}.npy"
            if not npy_path.exists():
                continue
            print(f"[fino] {cfg['exp']} seed={seed} ...", flush=True)
            r = analyze_one(cfg["exp"], seed, npy_path)
            print(f"  -> mejor_th_fino={r['best_threshold_fino']:.4f}  "
                  f"macro_f1_fino={r['macro_f1_fino']:.4f}  "
                  f"{'BORDE DEL GRID FINO -- revisar mas alla de 0.9999' if r['en_borde_del_grid_fino'] else ''}  "
                  f"forma={r['forma_curva']} (caida={r['caida_f1']:.4f})", flush=True)
            per_relation = r.pop("per_relation")
            for rel, m in per_relation.items():
                per_rel_rows.append({
                    "exp": r["exp"], "seed": r["seed"], "relation": rel,
                    "precision": m["precision"], "recall": m["recall"],
                    "f1": m["f1"], "support": m["support"],
                })
            rows.append(r)

    if not rows:
        print("Todavia no hay ningun blind_probs_*.npy guardado.")
        return

    df = pd.DataFrame(rows)
    out_dir = REPO / "outputs" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "multiseed_finegrid_resultados.csv", index=False)

    per_rel_df = pd.DataFrame(per_rel_rows)
    per_rel_df.to_csv(out_dir / "multiseed_finegrid_per_relation.csv", index=False)

    print("\n" + "=" * 70)
    print("=== F1 por relacion en el threshold OPTIMO, media +/- std entre seeds ===")
    per_rel_agg = (per_rel_df.groupby(["exp", "relation"])["f1"]
                   .agg(["mean", "std", "count"]).reset_index())
    pivot_rel = per_rel_agg.pivot(index="relation", columns="exp", values="mean")
    pivot_rel["media_todos_encoders"] = pivot_rel.mean(axis=1)
    pivot_rel = pivot_rel.sort_values("media_todos_encoders")
    print(pivot_rel.round(3).to_string())
    print(f"\nGuardado: outputs/multiseed/multiseed_finegrid_per_relation.csv "
          f"(P/R/F1/soporte por encoder x seed x relacion, en el threshold optimo de cada uno)")

    print("\n" + "=" * 70)
    print(f"=== Media +/- std por encoder (fino, n={df.groupby('exp').size().to_dict()}) ===")
    agg = df.groupby("exp")["macro_f1_fino"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(agg.to_string())

    print("\n=== Ranking seed a seed (fino) ===")
    pivot = df.pivot(index="seed", columns="exp", values="macro_f1_fino")
    print(pivot.to_string())
    if pivot.shape[1] > 1 and not pivot.isnull().values.any():
        rankings = pivot.rank(axis=1, ascending=False)
        stable = (rankings.nunique() == 1).all()
        print(f"\n¿Ranking identico en todas las seeds disponibles? {'SI' if stable else 'NO'}")

        print("\n=== Diferencias pareadas por seed (col - fila) ===")
        exps = pivot.columns.tolist()
        for i, a in enumerate(exps):
            for b in exps[i + 1:]:
                diffs = pivot[b] - pivot[a]
                signo = ("SIEMPRE +" if (diffs > 0).all() else
                         "SIEMPRE -" if (diffs < 0).all() else "CAMBIA DE SIGNO")
                print(f"{b} - {a}: {diffs.values.round(4).tolist()}  -> {signo}")
    else:
        print("(faltan combos para comparar ranking completo todavia)")

    print("\n=== Forma de curva por encoder ===")
    print(df.groupby("exp")["forma_curva"].value_counts().to_string())

    print(f"\nGuardado: outputs/multiseed/multiseed_finegrid_resultados.csv")


if __name__ == "__main__":
    main()

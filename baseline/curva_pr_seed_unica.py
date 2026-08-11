"""
Curva PR sin promediar entre seeds -- una sola seed (misma para los 4
encoders) para poder ver la curva real de un run concreto, en vez de la
media suavizada de multiseed_figuras_pr.py.

Usa los mismos CSV que multiseed_figuras_pr.py
(outputs/multiseed/multiseed_finegrid_pr.csv + multiseed_coarse_pr.csv),
pero en vez de agrupar y promediar por threshold entre las 3 seeds, filtra
directamente a --seed (por defecto 42) y traza esa curva tal cual.

Uso: python baseline/curva_pr_seed_unica.py [--seed 42]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "multiseed"
DATA_DIR = REPO / "data" / "english"

ENCODERS = [
    ("pubmedbert", "PubMedBERT"),
    ("biolinkbert_base", "BioLinkBERT-base"),
    ("biobert", "BioBERT"),
    ("scibert", "SciBERT"),
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def auc_trapz(x, y):
    order = np.argsort(x)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(np.array(y)[order], np.array(x)[order]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                     help="Seed unica a usar para los 4 encoders (misma seed en todos, sin promediar).")
    args = ap.parse_args()

    df = pd.read_csv(OUT / "multiseed_finegrid_pr.csv")
    df_full = pd.concat(
        [pd.read_csv(OUT / "multiseed_coarse_pr.csv"), df],
        ignore_index=True,
    ).sort_values(["exp", "seed", "threshold"])

    available_seeds = sorted(df_full["seed"].unique())
    if args.seed not in available_seeds:
        raise SystemExit(f"seed {args.seed} no esta en el CSV. Disponibles: {available_seeds}")

    df_seed = df_full[df_full["seed"] == args.seed]

    # Mismos extremos triviales que en la version promediada: a threshold=0 se
    # predice todo el pool (recall=1, precision=prevalencia); a threshold=1 no
    # se predice nada (recall=0, precision=1 por convencion tipo sklearn).
    gold_dev = pd.read_csv(DATA_DIR / "eng-dev-rel.tsv", sep="\t")
    total_candidates = sum(1 for l in open(DATA_DIR / "eng_dev_blind.txt", encoding="utf-8") if l.strip())
    total_positives = gold_dev.drop_duplicates(subset=["document_id", "head_span", "tail_span"]).shape[0]
    prevalence = total_positives / total_candidates

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, (key, label) in enumerate(ENCODERS):
        color = COLORS[i % len(COLORS)]
        g = df_seed[df_seed["exp"] == key].sort_values("threshold")
        if g.empty:
            print(f"aviso: no hay datos para {key} en seed={args.seed}, se omite")
            continue
        # threshold->0 (predice todo) va PRIMERO: recall=1, precision=prevalencia.
        # threshold->1 (no predice nada) va AL FINAL: recall=0, precision=1.
        # Los datos reales van de threshold bajo (recall alto) a threshold alto
        # (recall bajo), asi que los extremos deben ir en ESTE orden para no
        # crear un salto hacia atras en el eje X (el bug de antes).
        recall = np.concatenate([[1.0], g["recall_bin"].values, [0.0]])
        precision = np.concatenate([[prevalence], g["precision_bin"].values, [1.0]])
        aucpr = auc_trapz(recall, precision)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{label} (AUC-PR={aucpr:.3f})")

    ax.set_xlabel("Recall (binario: relacion detectada vs no)")
    ax.set_ylabel("Precision (binario)")
    ax.set_title(f"Curva PR -- deteccion de relacion vs no_relation\n"
                 f"(seed={args.seed} unica, SIN promediar entre seeds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out_path = OUT / f"curva_pr_seed{args.seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"guardado: {out_path}")


if __name__ == "__main__":
    main()

"""
save_artifacts.py — Artefactos extra por experimento (BioNNE-R, track ingles).

Objetivo: guardar en la carpeta de cada experimento todo lo necesario para
recalcular metricas, hacer analisis de errores y tests de significancia entre
modelos SIN reentrenar ni reinferir. Basado en practicas de evaluacion de RE
biomedica (per-class F1, IC bootstrap, multi-seed/std, McNemar, reproducibilidad).

Uso tipico (en un notebook, tras cargar el checkpoint y tener pred_df de dev):

    import sys; sys.path.insert(0, "../baseline")
    from save_artifacts import dump_all_artifacts
    dump_all_artifacts(
        model_pred, encoder_pred, dev_instances, pred_df,
        rel2id, OUTPUT_DIR, EXPERIMENT_NAME, MODEL_NAME,
        hyperparams=dict(max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
                         learning_rate=LEARNING_RATE, epochs=EPOCHS,
                         warmup_steps=WARMUP_STEPS, neg_ratio=NEG_RATIO, seed=SEED),
        history=history, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
    )
"""
import os
import json
import platform
import subprocess
import datetime
import numpy as np


# --------------------------------------------------------------------------
# 2) Reproducibilidad: run_meta.json
# --------------------------------------------------------------------------
def _git_commit(repo_dir=".."):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _versions():
    v = {"python": platform.python_version()}
    try:
        import torch
        v["torch"] = torch.__version__
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import transformers
        v["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import opennre
        v["opennre"] = getattr(opennre, "__version__", "unknown")
    except Exception:
        pass
    return v


def save_run_meta(output_dir, exp_name, model_name, hyperparams,
                  n_params=None, best_epoch=None, best_macro_f1=None,
                  train_seconds=None, extra=None):
    meta = {
        "experiment": exp_name,
        "model": model_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "env": _versions(),
        "hyperparameters": hyperparams,
        "n_params": int(n_params) if n_params is not None else None,
        "best_epoch": best_epoch,
        "best_macro_f1_dev": best_macro_f1,
        "train_seconds": round(train_seconds, 1) if train_seconds else None,
    }
    if extra:
        meta.update(extra)
    p = os.path.join(str(output_dir), f"run_meta_{exp_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta]  run_meta -> {p}")
    return meta


# --------------------------------------------------------------------------
# 3) Probabilidades en DEV + gold (para recomputar metricas / umbral / tests)
# --------------------------------------------------------------------------
def infer_probs(model, encoder, instances, num_class, batch_size=16, device=None):
    """Distribucion softmax completa por instancia (misma logica que el blind)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    all_probs = np.zeros((len(instances), num_class), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(instances), batch_size):
            batch = instances[s:s + batch_size]
            tok = [encoder.tokenize({"text": i["text"],
                                     "h": {"pos": i["h"]["pos"]},
                                     "t": {"pos": i["t"]["pos"]}}) for i in batch]
            fields = [torch.cat([t[k] for t in tok], dim=0).to(device)
                      for k in range(len(tok[0]))]
            logits = model(*fields)
            all_probs[s:s + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
    return all_probs


def save_probs(output_dir, exp_name, split, probs, gold_ids):
    pp = os.path.join(str(output_dir), f"{split}_probs_{exp_name}.npy")
    gp = os.path.join(str(output_dir), f"{split}_gold_{exp_name}.npy")
    np.save(pp, probs)
    np.save(gp, np.asarray(gold_ids))
    print(f"[probs] {split}_probs {probs.shape} -> {pp}")
    print(f"[probs] {split}_gold  -> {gp}")


# --------------------------------------------------------------------------
# 4) Predicciones por instancia enriquecidas (analisis estratificado)
# --------------------------------------------------------------------------
def save_detailed_predictions(output_dir, exp_name, split, pred_df):
    """pred_df debe incluir gold, head_type, tail_type, spans, score."""
    p = os.path.join(str(output_dir), f"pred_{split}_detallado_{exp_name}.tsv")
    pred_df.to_csv(p, sep="\t", index=False)
    print(f"[preds] detalladas ({len(pred_df)} filas) -> {p}")
    return p


# --------------------------------------------------------------------------
# 5) Matriz de confusion como datos (csv), no solo png
# --------------------------------------------------------------------------
def save_confusion_csv(output_dir, exp_name, y_true, y_pred, labels):
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    p = os.path.join(str(output_dir), f"confusion_{exp_name}.csv")
    df.to_csv(p)
    print(f"[conf]  matriz -> {p}")
    return df


# --------------------------------------------------------------------------
# 6a) IC bootstrap sobre macro-F1 (barra de error con un solo run)
# --------------------------------------------------------------------------
def bootstrap_macro_f1_ci(y_true, y_pred, labels=None, n_boot=1000, seed=42,
                          exclude=("no_relation",), output_dir=None, exp_name=None):
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lab = [l for l in (labels if labels is not None else sorted(set(y_true)))
           if l not in exclude]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))

    def mf1(a, b):
        return f1_score(a, b, labels=lab, average="macro", zero_division=0)

    point = float(mf1(y_true, y_pred))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = mf1(y_true[s], y_pred[s])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = {"macro_f1": point, "ci95_low": float(lo), "ci95_high": float(hi),
           "std_boot": float(boots.std()), "n_boot": n_boot, "seed": seed,
           "labels_evaluadas": lab}
    if output_dir and exp_name:
        p = os.path.join(str(output_dir), f"bootstrap_ci_{exp_name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[ci]    macro-F1={point:.4f}  IC95%=[{lo:.4f}, {hi:.4f}] -> {p}")
    return res


# --------------------------------------------------------------------------
# 6b) McNemar entre dos modelos (usar en el notebook de comparacion final)
# --------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    """Test de McNemar sobre aciertos por instancia de dos modelos.
    Devuelve n01/n10 (discordancias) y p-valor exacto (binomial)."""
    from scipy.stats import binomtest
    y_true = np.asarray(y_true)
    ca = (np.asarray(pred_a) == y_true)
    cb = (np.asarray(pred_b) == y_true)
    n01 = int(np.sum(~ca & cb))   # A falla, B acierta
    n10 = int(np.sum(ca & ~cb))   # A acierta, B falla
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    return {"n01_soloB_acierta": n01, "n10_soloA_acierta": n10,
            "discordantes": n, "p_value": float(p),
            "significativo_0.05": bool(p < 0.05)}


# --------------------------------------------------------------------------
# Orquestador: guarda 2-6 de una vez
# --------------------------------------------------------------------------
def dump_all_artifacts(model, encoder, dev_instances, pred_df, rel2id,
                       output_dir, exp_name, model_name, hyperparams,
                       history=None, max_length=256, batch_size=16):
    print("=" * 60)
    print(f"ARTEFACTOS EXTRA: {exp_name}")
    print("=" * 60)
    labels = list(rel2id.keys())
    y_true = pred_df["gold"].tolist()
    y_pred = pred_df["relation"].tolist()

    # 2) reproducibilidad
    n_params = sum(p.numel() for p in model.parameters())
    best_epoch = best_macro = None
    if history:
        best = max(history, key=lambda h: h.get("val_macro_f1", 0))
        best_epoch = best.get("epoch")
        best_macro = best.get("val_macro_f1")
    save_run_meta(output_dir, exp_name, model_name, hyperparams,
                  n_params=n_params, best_epoch=best_epoch, best_macro_f1=best_macro)

    # 3) probs + gold en DEV
    dev_probs = infer_probs(model, encoder, dev_instances, len(rel2id),
                            batch_size=batch_size)
    gold_ids = [rel2id[i["relation"]] for i in dev_instances]
    save_probs(output_dir, exp_name, "dev", dev_probs, gold_ids)

    # 4) predicciones detalladas
    save_detailed_predictions(output_dir, exp_name, "dev", pred_df)

    # 5) confusion como csv
    save_confusion_csv(output_dir, exp_name, y_true, y_pred, labels)

    # 6a) IC bootstrap
    ci = bootstrap_macro_f1_ci(y_true, y_pred, labels=labels,
                               output_dir=output_dir, exp_name=exp_name)
    print("=" * 60)
    print("Artefactos extra guardados. Nada de esto requiere reentrenar.")
    return ci

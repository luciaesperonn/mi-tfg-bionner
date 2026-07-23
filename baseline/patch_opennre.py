#!/usr/bin/env python3
"""
Patch OpenNRE for compatibility with modern transformers and Windows.

Run once after installing OpenNRE:
    python patch_opennre.py

Fixes:
1. UTF-8 encoding for data file loading (Windows defaults to cp1251/cp1252)
2. AdamW import (removed from transformers >=4.x, use torch.optim.AdamW)
3. num_workers=0 (avoids CUDA multiprocessing OOM on Windows)

`add_macro_f1_metric()` is a separate, in-memory monkeypatch (not a file
patch): import it and call it from a notebook, after `import opennre`, to
make `dataset.eval()` also return `macro_f1` so `train_model(metric="macro_f1")`
can select checkpoints by Macro F1 instead of the OpenNRE default (micro_f1).
"""

import importlib
import re
from collections import Counter
from pathlib import Path


def patch():
    import opennre.framework.data_loader as dl
    import opennre.framework.sentence_re as sre

    dl_path = Path(dl.__file__)
    sre_path = Path(sre.__file__)

    patched = []

    # --- data_loader.py ---
    dl_text = dl_path.read_text(encoding="utf-8")
    dl_orig = dl_text

    # Fix 1: encoding='utf-8' on all open(path) calls
    dl_text = dl_text.replace("f = open(path)", "f = open(path, encoding='utf-8')")

    # Fix 3: num_workers default 8 -> 0
    dl_text = dl_text.replace("num_workers=8", "num_workers=0")

    if dl_text != dl_orig:
        dl_path.write_text(dl_text, encoding="utf-8")
        patched.append(str(dl_path))

    # --- sentence_re.py ---
    sre_text = sre_path.read_text(encoding="utf-8")
    sre_orig = sre_text

    # Fix 2: Replace transformers.AdamW with torch.optim.AdamW
    sre_text = sre_text.replace(
        "from transformers import AdamW",
        "from torch.optim import AdamW",
    )
    # Remove correct_bias kwarg (not supported by torch AdamW)
    sre_text = sre_text.replace(
        "self.optimizer = AdamW(grouped_params, correct_bias=False)",
        "self.optimizer = AdamW(grouped_params)",
    )

    if sre_text != sre_orig:
        sre_path.write_text(sre_text, encoding="utf-8")
        patched.append(str(sre_path))

    if patched:
        print(f"Patched {len(patched)} file(s):")
        for p in patched:
            print(f"  {p}")
    else:
        print("Already patched, nothing to do.")


def add_macro_f1_metric():
    """Monkeypatch SentenceREDataset.eval to also return 'macro_f1'.

    OpenNRE's dataset.eval() only computes acc/micro_p/micro_r/micro_f1, and
    train_model() can only select the best checkpoint by a key present in
    that dict. This wraps it to additionally compute Macro F1 over all
    relations except the negative class (no_relation/NA/Other), which is the
    metric reported throughout this TFG.
    """
    import opennre.framework.data_loader as dl

    orig_eval = dl.SentenceREDataset.eval

    def _eval_with_macro(self, pred_result, use_name=False):
        result = orig_eval(self, pred_result, use_name=use_name)

        neg_names = {"NA", "na", "no_relation", "Other", "Others"}
        id2rel_local = {v: k for k, v in self.rel2id.items()}

        gold = [
            (self.data[i]["relation"] if use_name else self.rel2id[self.data[i]["relation"]])
            for i in range(len(self.data))
        ]

        tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
        for g, p in zip(gold, pred_result):
            support[g] += 1
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

        f1s = []
        for key, n_support in support.items():
            name = key if use_name else id2rel_local[key]
            if name in neg_names or n_support == 0:
                continue
            p_ = tp[key] / (tp[key] + fp[key]) if (tp[key] + fp[key]) else 0
            r_ = tp[key] / (tp[key] + fn[key]) if (tp[key] + fn[key]) else 0
            f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0
            f1s.append(f1)

        result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
        return result

    dl.SentenceREDataset.eval = _eval_with_macro


if __name__ == "__main__":
    patch()

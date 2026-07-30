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


def fix_large_model_hidden_size():
    """Monkeypatch BERTEntityEncoder.__init__ to size self.linear from the
    actual loaded model's hidden_size instead of a hardcoded 768*2.

    Stock OpenNRE hardcodes `self.hidden_size = 768 * 2`, which matches
    "base"-sized backbones (hidden_size=768) but breaks any "large" backbone
    (hidden_size=1024, e.g. BioLinkBERT-large, BiomedBERT-large-uncased):
    forward() concatenates head+tail hidden states into a (B, 2*1024) tensor,
    but self.linear was built as (1536, 1536), so the first forward pass
    raises "mat1 and mat2 shapes cannot be multiplied (Bx2048 and 1536x1536)".
    """
    from opennre.encoder.bert_encoder import BERTEntityEncoder
    from transformers import BertModel, BertTokenizer
    from torch import nn

    def _patched_init(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.hidden_size = self.bert.config.hidden_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    BERTEntityEncoder.__init__ = _patched_init


def fix_nested_entity_tokenize():
    """Monkeypatch BERTEntityEncoder.tokenize to handle head/tail entities
    that overlap or are nested in one another.

    Stock OpenNRE's tokenize() assumes head and tail never overlap: it splits
    the sentence into sent0+ent0+sent1+ent1+sent2 assuming pos_min (the
    entity that starts first) ENDS before pos_max (the other entity) STARTS.
    When one entity is nested inside the other (e.g. "lung" inside "lung
    injury", very common here for relations like AFFECTS/PART_OF/SUBCLASS_OF
    between a specific term and the general term it contains), that
    assumption breaks silently: sent1 becomes an empty/negative slice, and
    the trailing text after the inner entity gets duplicated into ent1/sent2
    while also still being part of ent0. The model ends up seeing token
    sequences with repeated words and misplaced [unused] markers instead of
    the real sentence.

    Measured impact on this project's data: only ~1.6% of the full blind
    candidate pool has overlapping spans (mostly negatives), but ~51% of the
    actual gold relations in the blind set do -- nesting correlates with
    being a true positive, since it's exactly the "specific-in-general" shape
    that relations like AFFECTS/PART_OF/SUBCLASS_OF have. ~13% of train is
    affected too.

    Fix: instead of assuming exactly 3 non-overlapping regions, cut the
    sentence at the union of all 4 entity-boundary offsets, tokenize each
    resulting segment once, and insert the [unused0-3] markers at the cut
    points where each entity opens/closes (closes before opens on ties, so a
    marker is never split by another opening exactly on its boundary). For
    the non-overlapping case this reduces to exactly the same segmentation
    as the stock code (verified byte-for-byte identical on real examples);
    for the nested/overlapping case it produces the same tokens as tokenizing
    the whole sentence once, with the markers correctly placed around each
    entity and no duplicated or dropped words.
    """
    from opennre.encoder.bert_encoder import BERTEntityEncoder
    import torch

    def _patched_tokenize(self, item):
        is_token = "text" not in item
        sentence = item["token"] if is_token else item["text"]
        h_start, h_end = item["h"]["pos"]
        t_start, t_end = item["t"]["pos"]

        def tok(s):
            s = " ".join(s) if is_token else s
            return self.tokenizer.tokenize(s) if s else []

        def sl(a, b):
            return sentence[a:b]

        n = len(sentence)
        cuts = sorted(set([0, h_start, h_end, t_start, t_end, n]))

        opens_at, closes_at = {}, {}
        opens_at.setdefault(h_start, []).append("h")
        opens_at.setdefault(t_start, []).append("t")
        closes_at.setdefault(h_end, []).append("h")
        closes_at.setdefault(t_end, []).append("t")

        if self.mask_entity:
            head_open, head_close = ["[unused4]"], []
            tail_open, tail_close = ["[unused5]"], []
        else:
            head_open, head_close = ["[unused0]"], ["[unused1]"]
            tail_open, tail_close = ["[unused2]"], ["[unused3]"]

        re_tokens = ["[CLS]"]
        pos1 = pos2 = None
        for i in range(len(cuts) - 1):
            a, b = cuts[i], cuts[i + 1]
            for who in closes_at.get(a, []):
                re_tokens += head_close if who == "h" else tail_close
            for who in opens_at.get(a, []):
                if who == "h":
                    pos1 = len(re_tokens); re_tokens += head_open
                else:
                    pos2 = len(re_tokens); re_tokens += tail_open
            if b > a:
                re_tokens += tok(sl(a, b))
        for who in closes_at.get(cuts[-1], []):
            re_tokens += head_close if who == "h" else tail_close
        re_tokens.append("[SEP]")

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2

    BERTEntityEncoder.tokenize = _patched_tokenize


def fix_entity_markers():
    """Reemplaza los marcadores de entidad [unused0]-[unused3] de OpenNRE por
    4 special tokens NUEVOS y DEDICADOS ([E1],[/E1],[E2],[/E2]), y de paso
    incluye el fix de fix_nested_entity_tokenize() (mismo tokenize() nuevo).
    Supersede a fix_nested_entity_tokenize: llamar a esta en vez de esa si se
    quieren ambos arreglos (que es el caso siempre que importe la identidad
    del marcador, no solo el anidamiento).

    Motivo -- bug encontrado en outputs/2A-pubmedbert-typed vs 2B-scibert-typed:
    [unused0]-[unused3] son huecos reservados por BERT para justo este uso,
    pero varios tokenizers de dominio biomedico reconstruyen el vocabulario
    desde cero y esos huecos dejan de apuntar a slots realmente "en blanco":

        PubMedBERT:       [unused0..3] -> ids [1,1,1,1]      (=[UNK] los 4)
        BioLinkBERT-base: [unused0..3] -> ids [1,1,1,1]      (=[UNK] los 4)
        BioBERT:          [unused0..3] -> ids [100,1,2,3]    (el primero =[UNK])
        SciBERT:          [unused0..3] -> ids [1,2,3,4]      (los 4 distintos, [UNK]=101)

    Con los 4 marcadores colapsados al mismo id que "palabra desconocida", el
    modelo no puede distinguir "aqui empieza la cabeza" de "aqui termina la
    cola" ni de una palabra cualquiera fuera de vocabulario -- para 3 de los 4
    encoders de este TFG el mecanismo de marcado de entidades de
    BERTEntityEncoder estaba, en la practica, apagado.

    Fix: `tokenizer.add_special_tokens(...)` + `bert.resize_token_embeddings(...)`
    -- la forma estandar de anadir tokens nuevos a cualquier vocabulario,
    garantiza 4 ids propios y distintos sin importar que huecos tuviera el
    vocabulario original. Los embeddings nuevos se inicializan por
    `resize_token_embeddings` (normal multivariante con la media/covarianza de
    los embeddings existentes) y se entrenan desde ese punto de partida durante
    el fine-tuning normal.
    """
    from opennre.encoder.bert_encoder import BERTEntityEncoder
    from transformers import BertModel, BertTokenizer
    from torch import nn
    import torch

    HEAD_OPEN, HEAD_CLOSE = "[E1]", "[/E1]"
    TAIL_OPEN, TAIL_CLOSE = "[E2]", "[/E2]"
    MASK_HEAD, MASK_TAIL = "[E1-MASK]", "[E2-MASK]"

    def _patched_init(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": [HEAD_OPEN, HEAD_CLOSE, TAIL_OPEN, TAIL_CLOSE, MASK_HEAD, MASK_TAIL]
        })
        if added > 0:
            self.bert.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.bert.config.hidden_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def _patched_tokenize(self, item):
        is_token = "text" not in item
        sentence = item["token"] if is_token else item["text"]
        h_start, h_end = item["h"]["pos"]
        t_start, t_end = item["t"]["pos"]

        def tok(s):
            s = " ".join(s) if is_token else s
            return self.tokenizer.tokenize(s) if s else []

        n = len(sentence)
        cuts = sorted(set([0, h_start, h_end, t_start, t_end, n]))
        opens_at, closes_at = {}, {}
        opens_at.setdefault(h_start, []).append("h")
        opens_at.setdefault(t_start, []).append("t")
        closes_at.setdefault(h_end, []).append("h")
        closes_at.setdefault(t_end, []).append("t")

        if self.mask_entity:
            head_open, head_close = [MASK_HEAD], []
            tail_open, tail_close = [MASK_TAIL], []
        else:
            head_open, head_close = [HEAD_OPEN], [HEAD_CLOSE]
            tail_open, tail_close = [TAIL_OPEN], [TAIL_CLOSE]

        re_tokens = ["[CLS]"]
        pos1 = pos2 = None
        for i in range(len(cuts) - 1):
            a, b = cuts[i], cuts[i + 1]
            for who in closes_at.get(a, []):
                re_tokens += head_close if who == "h" else tail_close
            for who in opens_at.get(a, []):
                if who == "h":
                    pos1 = len(re_tokens); re_tokens += head_open
                else:
                    pos2 = len(re_tokens); re_tokens += tail_open
            if b > a:
                re_tokens += tok(sentence[a:b])
        for who in closes_at.get(cuts[-1], []):
            re_tokens += head_close if who == "h" else tail_close
        re_tokens.append("[SEP]")

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1
        return indexed_tokens, att_mask, pos1, pos2

    BERTEntityEncoder.__init__ = _patched_init
    BERTEntityEncoder.tokenize = _patched_tokenize


if __name__ == "__main__":
    patch()

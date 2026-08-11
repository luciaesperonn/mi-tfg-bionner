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

    # Fix 4: eval(line) -> json.loads(line). eval() treats each data line as a
    # Python literal, which happens to work for this project's original data
    # (no lowercase true/false/null ever appeared in it) but breaks the
    # moment a JSONL file has a real JSON boolean/null -- e.g. a "synthetic":
    # true field added by an augmentation script. json is already imported
    # in this module, so this is a drop-in replacement with no behavior
    # change on data that was eval-safe before.
    dl_text = dl_text.replace("self.data.append(eval(line))", "self.data.append(json.loads(line))")

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


def fix_universal_encoder():
    """Monkeypatch BERTEntityEncoder to work with any AutoModel/AutoTokenizer
    backbone, not just BERT -- in particular XLM-RoBERTa (SentencePiece).

    Supersedes fix_large_model_hidden_size(): also derives hidden_size from
    the loaded model's config, so it covers "large" backbones too.

    OpenNRE hardcodes BertModel + BertTokenizer + the literal strings
    '[CLS]'/'[SEP]' + pad id 0 + '[unused0]'-'[unused3]' entity markers.
    Every WordPiece/BERT-vocab backbone already used in this TFG (PubMedBERT,
    BioLinkBERT base/large, BioBERT, SciBERT, BiomedBERT-large) happens to
    satisfy all of those assumptions (cls_token='[CLS]', sep_token='[SEP]',
    pad_token_id=0), so for them this patch is behavior-preserving:
    AutoModel/AutoTokenizer resolve to the exact same classes, and
    tokenize() produces byte-identical output -- whatever pre-existing state
    the '[unused]' markers have per vocab (some collide with [UNK], see
    fix_entity_markers) is left exactly as before, so results already saved
    for those encoders stay reproducible.

    XLM-RoBERTa breaks every one of those assumptions: cls_token='<s>',
    sep_token='</s>', pad_token_id=1, and '[unused0]'-'[unused5]' don't
    exist in its SentencePiece vocab at all (they'd silently collapse to
    <unk>, and padding with the hardcoded 0 would inject spurious '<s>'
    tokens into the padded region). This patch detects any backbone like
    that (tokenizer doesn't already match the BERT convention) and takes
    the safe path validated in the original XLM-RoBERTa experiment: register
    '[unused0]'-'[unused5]' as real tokens with embeddings initialized to
    the MEAN of the existing embedding matrix (random init made the model
    collapse to predicting only 'no_relation'), and use the tokenizer's own
    cls/sep/pad tokens instead of the hardcoded BERT ones.
    """
    from opennre.encoder.bert_encoder import BERTEntityEncoder
    from transformers import AutoModel, AutoTokenizer
    from torch import nn
    import torch

    ENTITY_MARKERS = ["[unused0]", "[unused1]", "[unused2]",
                      "[unused3]", "[unused4]", "[unused5]"]

    def _patched_init(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=False)

        is_bert_vocab = (self.tokenizer.cls_token == "[CLS]"
                          and self.tokenizer.sep_token == "[SEP]"
                          and self.tokenizer.pad_token_id == 0)
        if not is_bert_vocab:
            new_tokens = [t for t in ENTITY_MARKERS if t not in self.tokenizer.get_vocab()]
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                self.bert.resize_token_embeddings(len(self.tokenizer))
                with torch.no_grad():
                    n_new = len(new_tokens)
                    emb = self.bert.get_input_embeddings().weight
                    mean = emb[:-n_new].mean(dim=0)
                    emb[-n_new:] = mean.unsqueeze(0).repeat(n_new, 1)

        self.hidden_size = self.bert.config.hidden_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def _patched_tokenize(self, item):
        is_token = "text" not in item
        sentence = item["token"] if is_token else item["text"]
        pos_head = item["h"]["pos"]
        pos_tail = item["t"]["pos"]

        if pos_head[0] > pos_tail[0]:
            pos_min, pos_max, rev = pos_tail, pos_head, True
        else:
            pos_min, pos_max, rev = pos_head, pos_tail, False

        def tok(s):
            s = " ".join(s) if is_token else s
            return self.tokenizer.tokenize(s) if s else []

        sent0 = tok(sentence[:pos_min[0]])
        ent0 = tok(sentence[pos_min[0]:pos_min[1]])
        sent1 = tok(sentence[pos_min[1]:pos_max[0]])
        ent1 = tok(sentence[pos_max[0]:pos_max[1]])
        sent2 = tok(sentence[pos_max[1]:])

        if self.mask_entity:
            ent0 = ["[unused4]"] if not rev else ["[unused5]"]
            ent1 = ["[unused5]"] if not rev else ["[unused4]"]
        else:
            ent0 = ["[unused0]"] + ent0 + ["[unused1]"] if not rev else ["[unused2]"] + ent0 + ["[unused3]"]
            ent1 = ["[unused2]"] + ent1 + ["[unused3]"] if not rev else ["[unused0]"] + ent1 + ["[unused1]"]

        re_tokens = [self.tokenizer.cls_token] + sent0 + ent0 + sent1 + ent1 + sent2 + [self.tokenizer.sep_token]

        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(pad_id)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2

    BERTEntityEncoder.__init__ = _patched_init
    BERTEntityEncoder.tokenize = _patched_tokenize


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


def fix_universal_encoder_with_markers():
    """Version backbone-agnostic de fix_entity_markers(): combina el soporte
    AutoModel/AutoTokenizer de fix_universal_encoder() con los marcadores
    dedicados y el tokenize() consciente de anidamiento de fix_entity_markers()
    (ver HALLAZGOS-BUGS-TOKENIZACION.md, Bug 1 y Bug 2).

    fix_universal_encoder() decide si reutilizar [unused0]-[unused5] mirando
    si cls/sep/pad "parecen" BERT (`cls_token=='[CLS]'`, `sep_token=='[SEP]'`,
    `pad_token_id==0`). Esa comprobacion no es suficiente: el tokenizer de
    DeBERTa-v3 la pasa (usa exactamente esos tokens) pero su vocabulario no
    reserva huecos [unused] de verdad -- `convert_tokens_to_ids('[unused0]')`
    devuelve el mismo id que `[UNK]` (comprobado: los 4 marcadores colapsan a
    id=3), el mismo fallo que Bug 2 documento para PubMedBERT/BioLinkBERT-base/
    BioBERT. "Parecer BERT" en cls/sep/pad no implica tener huecos [unused]
    reales.

    Por eso esta version SIEMPRE anade tokens nuevos y dedicados
    ([E1]/[/E1]/[E2]/[/E2] + variantes MASK) via `add_special_tokens`, sin
    importar el backbone -- evita depender de que el vocabulario tenga huecos
    libres, sea cual sea. Los embeddings nuevos se inicializan a la MEDIA de
    la matriz existente (no al init por defecto de `resize_token_embeddings`):
    es el criterio ya validado en `fix_universal_encoder()` para XLM-RoBERTa,
    donde el init por defecto colapsaba el modelo a predecir solo
    'no_relation'.

    Incluye tambien el tokenize() de `fix_nested_entity_tokenize()`/
    `fix_entity_markers()` (Bug 1): corta la frase por la union de los 4
    bordes de entidad en vez de asumir que head y tail nunca se solapan.

    Tambien parchea forward(): el original hace
    `hidden, _ = self.bert(..., return_dict=False)` asumiendo que el backbone
    siempre devuelve (last_hidden_state, pooler_output). Backbones sin capa de
    pooler (DebertaV2Model, entre otros) devuelven una tupla de un solo
    elemento con return_dict=False, y ese unpacking revienta con
    "not enough values to unpack". El fix toma `[0]` (siempre
    last_hidden_state, el unico dato que BERTEntityEncoder usa) en vez de
    asumir la longitud de la tupla.
    """
    from opennre.encoder.bert_encoder import BERTEntityEncoder
    from transformers import AutoModel, AutoTokenizer
    from torch import nn
    import torch

    HEAD_OPEN, HEAD_CLOSE = "[E1]", "[/E1]"
    TAIL_OPEN, TAIL_CLOSE = "[E2]", "[/E2]"
    MASK_HEAD, MASK_TAIL = "[E1-MASK]", "[E2-MASK]"
    NEW_TOKENS = [HEAD_OPEN, HEAD_CLOSE, TAIL_OPEN, TAIL_CLOSE, MASK_HEAD, MASK_TAIL]

    def _patched_init(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        # dtype=torch.float32 explicito: transformers>=5 carga cada checkpoint
        # en el dtype con el que se subio a HF Hub si no se especifica lo
        # contrario -- microsoft/deberta-v3-base esta en fp16. Entrenar fp16
        # con un optimizer/clip_grad_norm_ pensados para fp32 (sin loss
        # scaling, como el resto de este pipeline) diverge a NaN en el primer
        # optimizer step (comprobado: loss=nan ya en el segundo batch).
        self.bert = AutoModel.from_pretrained(pretrain_path, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=False)

        new_tokens = [t for t in NEW_TOKENS if t not in self.tokenizer.get_vocab()]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            self.bert.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                n_new = len(new_tokens)
                emb = self.bert.get_input_embeddings().weight
                mean = emb[:-n_new].mean(dim=0)
                emb[-n_new:] = mean.unsqueeze(0).repeat(n_new, 1)

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

        re_tokens = [self.tokenizer.cls_token]
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
        re_tokens.append(self.tokenizer.sep_token)

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(pad_id)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2

    def _patched_forward(self, token, att_mask, pos1, pos2):
        hidden = self.bert(token, attention_mask=att_mask, return_dict=False)[0]
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)
        x = torch.cat([head_hidden, tail_hidden], 1)
        x = self.linear(x)
        return x

    BERTEntityEncoder.__init__ = _patched_init
    BERTEntityEncoder.tokenize = _patched_tokenize
    BERTEntityEncoder.forward = _patched_forward


if __name__ == "__main__":
    patch()

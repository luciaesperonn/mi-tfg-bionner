"""
Intento 6 de augmentation para ALTERNATIVE_NAME: distinto de 5A-5D en dos
cosas a la vez (no solo "generar mas del mismo tipo"):

1. Pares LIMPIOS de verdad: los 47 "pares nuevos" de 5B/5D incluian 30/47
   con solapamiento lexico (muscle/muscular, gland/glandular...) -- por
   definicion del propio informe (seccion 9), esos son del subtipo "facil"
   que el modelo ya domina, así que 5B/5D nunca aislo limpiamente la
   hipotesis de la seccion 9 (targeting subtipo de bajo solapamiento).
   Aqui se usan SOLO los 17 pares de 5B/5D que de verdad no comparten raiz
   (kidney/renal, liver/hepatic...) + 27 pares nuevos verificados
   (heart attack/myocardial infarction, stroke/cerebrovascular accident...),
   verificados por separado contra train+aug+blind-gold para que no haya
   fuga de datos ni pares repetidos.

2. Hard negatives emparejados: por cada uno de los 44 pares, ademas de la
   frase positiva (equivalencia explicita), se genera una frase NEGATIVA
   donde los dos terminos co-ocurren en la misma frase pero SIN afirmar que
   son lo mismo (p.ej. uno es hallazgo asociado al otro, o se miden por
   separado) -- etiquetada no_relation. Esto ataca directamente el fallo
   diagnosticado en 5B (secc. 4.5.3/9.4): el modelo ensancha el criterio de
   "esto es ALTERNATIVE_NAME" a cualquier co-ocurrencia end vez de aprender
   la frontera fina. Sin negativos duros emparejados, mas positivos solo
   pueden ensanchar el criterio, nunca afinarlo.

Requiere GEMINI_API_KEY. Uso: python augment_attempt6.py
"""
import json
import time
from pathlib import Path

from google import genai
from google.genai import types
import sys
sys.path.insert(0, ".")
from augment_llm import call_gemini, strip_markers, parse_batch_response  # noqa: E402

MODEL = "gemini-3.5-flash"
BATCH_SIZE = 20

NEW_PAIRS_27 = [
    ("heart attack", "myocardial infarction"), ("stroke", "cerebrovascular accident"),
    ("swelling", "edema"), ("itching", "pruritus"), ("bruising", "ecchymosis"),
    ("nosebleed", "epistaxis"), ("fainting", "syncope"), ("shortness of breath", "dyspnea"),
    ("difficulty swallowing", "dysphagia"), ("yellow skin", "jaundice"),
    ("excessive thirst", "polydipsia"), ("bedsore", "pressure ulcer"),
    ("ringing in the ears", "tinnitus"), ("pinkeye", "conjunctivitis"),
    ("nearsightedness", "myopia"), ("farsightedness", "hyperopia"),
    ("lazy eye", "amblyopia"), ("clubfoot", "talipes"),
    ("whooping cough", "pertussis"), ("chickenpox", "varicella"),
    ("German measles", "rubella"), ("mumps", "parotitis"),
    ("water on the brain", "hydrocephalus"), ("hardening of the arteries", "atherosclerosis"),
    ("hunchback", "kyphosis"), ("knock-knees", "genu valgum"), ("bowlegs", "genu varum"),
]

CLEAN_17 = [
    ("kidney", "renal"), ("liver", "hepatic"), ("brain", "cerebral"), ("stomach", "gastric"),
    ("skin", "cutaneous"), ("bone", "osseous"), ("nerve", "neural"), ("blood", "hematologic"),
    ("vein", "venous"), ("tooth", "dental"), ("mouth", "oral"), ("nose", "nasal"),
    ("throat", "pharyngeal"), ("tongue", "lingual"), ("rib", "costal"), ("joint", "articular"),
    ("bladder", "vesical"),
]
ALL_44 = CLEAN_17 + NEW_PAIRS_27

POS_SYSTEM_PROMPT = """You are a biomedical relation-extraction data augmentation \
assistant. You will receive a numbered list of items, each giving two terms \
that are VERIFIED lay-term/clinical-term (or equivalent) synonym pairs for \
the same underlying concept. This fact is already confirmed.

For each item, write ONE new, realistic sentence in the style of a \
biomedical research abstract or clinical note that EXPLICITLY asserts the \
two terms refer to the same underlying concept. The sentence MUST contain \
an explicit equivalence marker -- e.g. "also known as", "also referred to \
as", "i.e.", "that is", "in other words", "synonymous with", "commonly \
called", "clinically termed" -- connecting the two terms directly. Do NOT \
use language that merely associates or co-locates the two terms.

Rules:
- Use both terms character-for-character exactly as given (same \
capitalization, no pluralizing/inflecting differently).
- Wrap the first term in [E1] and [/E1], the second in [E2] and [/E2], \
exactly once each.
- Vary sentence structure across items.

Output ONLY a JSON array of strings, one per item, same order as input, \
no markdown fences, no preamble."""

NEG_SYSTEM_PROMPT = """You are a biomedical relation-extraction data \
augmentation assistant. You will receive a numbered list of items, each \
giving two related biomedical terms (they are true synonyms of each other, \
but you must NOT reveal or use that fact here).

For each item, write ONE realistic biomedical-abstract-style sentence where \
BOTH terms appear, but the sentence must NOT assert or imply they are the \
same concept. Instead, use them as two separate, merely CO-OCCURRING or \
ASSOCIATED elements -- e.g. one is a risk factor, consequence, comorbidity, \
or separately measured finding relative to the other -- so that a careful \
reader would NOT conclude they are alternative names for the same thing. \
Do NOT use any equivalence marker ("also known as", "i.e.", "that is", \
"synonymous with", etc.) or any phrasing that suggests identity.

Rules:
- Use both terms character-for-character exactly as given.
- Wrap the first term in [E1] and [/E1], the second in [E2] and [/E2], \
exactly once each.
- Vary sentence structure and the type of non-identity relationship used \
across items.

Output ONLY a JSON array of strings, one per item, same order as input, \
no markdown fences, no preamble."""

ITEM_TEMPLATE = "{idx}. Term 1: {t1}\n   Term 2: {t2}"


def call_batch(client, pairs, system_prompt):
    items = [ITEM_TEMPLATE.format(idx=i, t1=t1, t2=t2) for i, (t1, t2) in enumerate(pairs, 1)]
    prompt = "\n\n".join(items)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt, max_output_tokens=4096, temperature=0.7,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            raw_text = response.text
            break
        except Exception as e:
            if attempt == 4:
                print(f"  aviso: fallo tras 5 intentos ({e})")
                return []
            time.sleep(min(2 ** attempt, 30))
    arr = parse_batch_response(raw_text, len(pairs))
    results = []
    for (t1, t2), raw_item in zip(pairs, arr):
        if not isinstance(raw_item, str):
            continue
        parsed = strip_markers(raw_item.strip())
        if parsed is None:
            continue
        plain, h_span, t_span = parsed
        if plain[h_span[0]:h_span[1]].strip().lower() != t1.strip().lower():
            continue
        if plain[t_span[0]:t_span[1]].strip().lower() != t2.strip().lower():
            continue
        results.append((plain, h_span, t_span, t1, t2))
    return results


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    client = genai.Client()
    out_pos = Path("../data/english/eng_train_llmaug_attempt6_pos.txt")
    out_neg = Path("../data/english/eng_train_llmaug_attempt6_neg.txt")

    # --- positivos: solo los 27 pares nuevos (los 17 limpios se reusan tal cual) ---
    print(f"Generando POSITIVOS para {len(NEW_PAIRS_27)} pares nuevos...")
    written_pos = 0
    with open(out_pos, "w", encoding="utf-8") as f:
        for i, batch in enumerate(chunked(NEW_PAIRS_27, BATCH_SIZE)):
            for plain, h_span, t_span, t1, t2 in call_batch(client, batch, POS_SYSTEM_PROMPT):
                inst = {"text": plain, "h": {"name": t1, "pos": list(h_span)},
                        "t": {"name": t2, "pos": list(t_span)}, "relation": "ALTERNATIVE_NAME",
                        "doc_id": f"llmaug6_pos_{t1}_{t2}".replace(" ", "_"),
                        "head_span": "0-0", "tail_span": "0-0",
                        "head_type": "DISO", "tail_type": "DISO"}
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                f.flush()
                written_pos += 1
            print(f"  lote {i+1}, {written_pos} generados hasta ahora", flush=True)
            time.sleep(2)
    print(f"Positivos nuevos: {written_pos}/{len(NEW_PAIRS_27)}\n")

    # --- hard negatives: los 44 pares (17 limpios + 27 nuevos) ---
    print(f"Generando HARD NEGATIVES para los {len(ALL_44)} pares...")
    written_neg = 0
    with open(out_neg, "w", encoding="utf-8") as f:
        for i, batch in enumerate(chunked(ALL_44, BATCH_SIZE)):
            for plain, h_span, t_span, t1, t2 in call_batch(client, batch, NEG_SYSTEM_PROMPT):
                inst = {"text": plain, "h": {"name": t1, "pos": list(h_span)},
                        "t": {"name": t2, "pos": list(t_span)}, "relation": "no_relation",
                        "doc_id": f"llmaug6_neg_{t1}_{t2}".replace(" ", "_"),
                        "head_span": "0-0", "tail_span": "0-0",
                        "head_type": "DISO", "tail_type": "DISO"}
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                f.flush()
                written_neg += 1
            print(f"  lote {i+1}, {written_neg} generados hasta ahora", flush=True)
            time.sleep(2)
    print(f"Hard negatives: {written_neg}/{len(ALL_44)}")


if __name__ == "__main__":
    main()

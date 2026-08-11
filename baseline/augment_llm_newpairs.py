"""Data augmentation de ALTERNATIVE_NAME con parejas de entidad NUEVAS.

Complementa a `augment_llm.py` (que parafrasea instancias EXISTENTES,
manteniendo las mismas ~95 parejas de entidad de siempre). Este script ataca
el problema distinto que se diagnostico tras el experimento 5A: parafrasear
no ayuda a `ALTERNATIVE_NAME` porque nunca introduce parejas de entidad
nuevas -- el modelo sigue sin ver el patron aplicado a entidades que no
conocia. Ver DATA-AUGMENTATION-RESUMEN-COMPLETO.md.

Las parejas nuevas (`NEW_PAIRS` mas abajo) NO las inventa el LLM -- son
vocabulario medico estandar, verificable en cualquier diccionario medico:
- Anatomico sustantivo<->adjetivo (kidney/renal, liver/hepatic...)
- Variantes morfologicas de la misma raiz (infection/infected,
  hypertension/hypertensive...)

Se comprobo antes de escribir esto que ninguna de las 47 parejas coincide
(en ningun orden) con las que ya hay en train + train_llmaug.

El LLM solo escribe la FRASE alrededor de la pareja ya verificada (con los
mismos marcadores [E1]/[E2] y el mismo filtro de coincidencia exacta que
augment_llm.py) -- no decide que la relacion es cierta, eso ya esta fijado
de antemano por la lista.

Uso:
    python augment_llm_newpairs.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

from google import genai

sys.path.insert(0, ".")
from augment_llm import call_gemini, strip_markers, parse_batch_response  # noqa: E402

BATCH_SIZE = 20

SYSTEM_PROMPT = """You are a biomedical relation-extraction data augmentation \
assistant. You will receive a numbered list of items. Each item gives two \
biomedical terms that are VERIFIED, well-established alternative names for \
the same concept (an anatomical noun/adjective pair, or a morphological \
variant of the same root -- e.g. "kidney"/"renal", "infection"/"infected"). \
This fact is already confirmed; you do not need to judge whether it is true.

For each item, write ONE new, realistic sentence in the style of a \
biomedical research abstract that EXPLICITLY asserts the two terms refer \
to the same underlying concept. The sentence MUST contain an explicit \
equivalence marker -- e.g. "also known as", "also referred to as", "i.e.", \
"that is", "in other words", "synonymous with", "the adjectival form of", \
"the term X is used interchangeably with Y" -- connecting the two terms \
directly.

It is NOT enough to use both terms in nearby clauses about the same organ \
or condition -- that only shows they are topically related, not that they \
are the same concept renamed. For example, for "kidney"/"renal", do NOT \
write something like "ultrasound of the kidney revealed a renal cell \
carcinoma" (renal cell carcinoma is a diagnosis found IN the kidney, not \
another name FOR the kidney) -- instead write something like "the kidney, \
also referred to using the adjectival form renal, showed structural \
changes" or "renal function, i.e. the physiological performance of the \
kidney, was assessed".

Rules (apply to every item):
- Use both terms character-for-character exactly as given -- do not change \
capitalization, pluralize, or inflect them differently.
- Wrap the first term in [E1] and [/E1], and the second term in [E2] and \
[/E2], exactly once each, around the verbatim text.
- Vary sentence structure and clinical context across items -- do not reuse \
the same template -- but every sentence must keep an explicit equivalence \
marker of some kind.

Output format:
- Return ONLY a JSON array of strings, one per item, in the SAME ORDER as \
the input list. Each string is the sentence with its [E1]/[E2] markers.
- The array must have exactly as many elements as there were input items.
- No explanation, no markdown code fences, no preamble -- output must start \
with `[` and end with `]`."""

ITEM_TEMPLATE = "{idx}. Term 1: {term1}\n   Term 2: {term2}"

# (termino1, termino2, head_type, tail_type) -- verificados antes de correr el
# script (ver docstring): ninguno se solapa con parejas ya presentes en train.
NEW_PAIRS = [
    ("kidney", "renal", "ANATOMY"), ("liver", "hepatic", "ANATOMY"),
    ("brain", "cerebral", "ANATOMY"), ("stomach", "gastric", "ANATOMY"),
    ("skin", "cutaneous", "ANATOMY"), ("bone", "osseous", "ANATOMY"),
    ("nerve", "neural", "ANATOMY"), ("muscle", "muscular", "ANATOMY"),
    ("blood", "hematologic", "ANATOMY"), ("spleen", "splenic", "ANATOMY"),
    ("pancreas", "pancreatic", "ANATOMY"), ("intestine", "intestinal", "ANATOMY"),
    ("colon", "colonic", "ANATOMY"), ("uterus", "uterine", "ANATOMY"),
    ("ovary", "ovarian", "ANATOMY"), ("testis", "testicular", "ANATOMY"),
    ("prostate", "prostatic", "ANATOMY"), ("artery", "arterial", "ANATOMY"),
    ("vein", "venous", "ANATOMY"), ("tooth", "dental", "ANATOMY"),
    ("mouth", "oral", "ANATOMY"), ("nose", "nasal", "ANATOMY"),
    ("throat", "pharyngeal", "ANATOMY"), ("tongue", "lingual", "ANATOMY"),
    ("rib", "costal", "ANATOMY"), ("spine", "spinal", "ANATOMY"),
    ("joint", "articular", "ANATOMY"), ("tendon", "tendinous", "ANATOMY"),
    ("gland", "glandular", "ANATOMY"), ("bladder", "vesical", "ANATOMY"),
    ("infection", "infected", "DISO"), ("hypertension", "hypertensive", "DISO"),
    ("necrosis", "necrotic", "DISO"), ("fibrosis", "fibrotic", "DISO"),
    ("ischemia", "ischemic", "DISO"), ("edema", "edematous", "DISO"),
    ("hemorrhage", "hemorrhagic", "DISO"), ("thrombosis", "thrombotic", "DISO"),
    ("neoplasm", "neoplastic", "DISO"), ("cancer", "cancerous", "DISO"),
    ("allergy", "allergic", "DISO"), ("anemia", "anemic", "DISO"),
    ("obesity", "obese", "DISO"), ("toxicity", "toxic", "DISO"),
    ("infarct", "infarcted", "DISO"), ("atrophy", "atrophic", "DISO"),
    ("hypoxia", "hypoxic", "DISO"),
]


def build_batch_prompt(pairs):
    items = [ITEM_TEMPLATE.format(idx=i, term1=t1, term2=t2)
             for i, (t1, t2, _) in enumerate(pairs, 1)]
    return "\n\n".join(items)


def call_batch(client, pairs):
    config_prompt = build_batch_prompt(pairs)
    raw_text = call_gemini_with_system(client, config_prompt)
    arr = parse_batch_response(raw_text, len(pairs))

    results = []
    for (term1, term2, etype), raw_item in zip(pairs, arr):
        if not isinstance(raw_item, str):
            continue
        parsed = strip_markers(raw_item.strip())
        if parsed is None:
            continue
        plain, h_span, t_span = parsed
        if plain[h_span[0]:h_span[1]].strip().lower() != term1.strip().lower():
            continue
        if plain[t_span[0]:t_span[1]].strip().lower() != term2.strip().lower():
            continue
        results.append({
            "text": plain,
            "h": {"name": term1, "pos": list(h_span)},
            "t": {"name": term2, "pos": list(t_span)},
            "relation": "ALTERNATIVE_NAME",
            "doc_id": f"llmaug_newpair_{term1}_{term2}".replace(" ", "_"),
            "head_span": "0-0",   # sin documento real de origen -- no aplica
            "tail_span": "0-0",
            "head_type": etype,
            "tail_type": etype,
        })
    return results


def call_gemini_with_system(client, prompt):
    """Como call_gemini() de augment_llm.py pero con el SYSTEM_PROMPT de este script."""
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4096,
        temperature=0.7,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    import time
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash", contents=prompt, config=config,
            )
            return response.text
        except Exception as e:
            if attempt == 4:
                print(f"  aviso: fallo tras 5 intentos ({e})")
                return None
            time.sleep(min(2 ** attempt, 30))
    return None


def main():
    client = genai.Client()
    out_path = Path("../data/english/eng_train_llmaug_newpairs.txt")

    print(f"{len(NEW_PAIRS)} parejas nuevas (verificadas, no inventadas por el LLM)")
    batches = [NEW_PAIRS[i:i + BATCH_SIZE] for i in range(0, len(NEW_PAIRS), BATCH_SIZE)]
    print(f"-> {len(batches)} llamadas de hasta {BATCH_SIZE} parejas cada una")

    written = 0
    kept_by_type = Counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, batch in enumerate(batches):
            for new_inst in call_batch(client, batch):
                f.write(json.dumps(new_inst, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                kept_by_type[new_inst["head_type"]] += 1
            print(f"  lote {i + 1}/{len(batches)}, {written} generadas hasta ahora", flush=True)

    print(f"\nTotal generado: {written}/{len(NEW_PAIRS)}")
    for t, n in kept_by_type.most_common():
        print(f"  {t}: {n}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()

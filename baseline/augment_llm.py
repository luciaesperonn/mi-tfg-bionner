"""Data augmentation de clases minoritarias con Gemini (capa gratuita).

Parafrasea instancias positivas ya existentes en train, manteniendo las dos
menciones de entidad EXACTAS (marcadas por el modelo con [E1]/[E2] para poder
recalcular sus offsets de caracter de forma determinista, sin fiarse de que
el LLM cuente caracteres) y la misma relacion. Las generaciones que no
reproducen ambas menciones tal cual se descartan -- es el filtro de calidad
frente al riesgo de ruido de etiqueta.

Deliberadamente NO genera frases desde cero (el LLM inventando head/tail
nuevos): eso reintroduce el problema de recalcular spans buscando el texto
en la frase (la fuente de los bugs de HALLAZGOS-BUGS-TOKENIZACION.md) y le
pide al LLM que invente un hecho biomedico en vez de parafrasear uno real.
Parafrasear con marcadores inline evita ambos riesgos a la vez.

Las peticiones van en LOTES (BATCH_SIZE instancias por llamada, cada una
pidiendo su propia frase parafraseada) para no chocar con la cuota diaria
baja de la capa gratuita (20 peticiones/dia en gemini-3.5-flash) -- antes
era una llamada por variante, ahora son BATCH_SIZE variantes por llamada.

Requiere GEMINI_API_KEY (gratis en https://aistudio.google.com/apikey) y
`pip install google-genai`.

Uso:
    python augment_llm.py --relations APPLIED_TO ALTERNATIVE_NAME \
        --variants-per-instance 2
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"
BATCH_SIZE = 20

SYSTEM_PROMPT = """You are a biomedical relation-extraction data augmentation \
assistant. You will receive a numbered list of items, each with a sentence \
and two entity mentions with the relation that holds between them. For each \
item, rewrite the sentence with different wording while preserving the same \
fact.

Rules (apply to every item):
- Keep both entity mentions character-for-character identical -- do not \
translate, abbreviate, expand, or pluralize them differently.
- The rewritten sentence must still clearly support the same relation \
between the same two entities -- do not weaken it into a vague or \
unrelated statement.
- If the relation is ALTERNATIVE_NAME, the sentence MUST contain an \
explicit marker that the two mentions refer to the SAME underlying \
concept -- e.g. "also known as", "also referred to as", "i.e.", "that is", \
"in other words", "synonymous with", "formerly called", "abbreviated as", \
or an equivalent explicit equivalence phrase. Do NOT use language that \
merely associates, sequences, or co-locates the two mentions without \
asserting they are the same thing -- phrases like "share features with", \
"progression into", "accompanied by", "represents a portion of", \
"complemented by", "tracked both X and Y" describe a DIFFERENT relation \
and must never be used for an ALTERNATIVE_NAME item, even if the original \
sentence hinted at them.
- Wrap the head entity in [E1] and [/E1], and the tail entity in [E2] and \
[/E2], exactly once each, around the verbatim entity text.

Output format:
- Return ONLY a JSON array of strings, one per item, in the SAME ORDER as \
the input list. Each string is the rewritten sentence with its [E1]/[E2] \
markers.
- The array must have exactly as many elements as there were input items.
- No explanation, no markdown code fences, no preamble -- output must start \
with `[` and end with `]`."""

ITEM_TEMPLATE = """{idx}. Sentence: {text}
   Head entity: {head}
   Tail entity: {tail}
   Relation: {relation}"""

MARKER_RE = re.compile(r"\[E1\](.*?)\[/E1\]|\[E2\](.*?)\[/E2\]", re.DOTALL)


def strip_markers(marked: str):
    """Quita los marcadores [E1]/[E2] y devuelve (texto_plano, h_span, t_span).

    Devuelve None si no aparecen ambos marcadores exactamente una vez.
    """
    h_span = t_span = None
    out = []
    cursor = 0
    for m in MARKER_RE.finditer(marked):
        out.append(marked[cursor:m.start()])
        start = sum(len(s) for s in out)
        if m.group(1) is not None:
            content = m.group(1)
            if h_span is not None:
                return None  # marcador duplicado -- descartar
            h_span = (start, start + len(content))
        else:
            content = m.group(2)
            if t_span is not None:
                return None
            t_span = (start, start + len(content))
        out.append(content)
        cursor = m.end()
    out.append(marked[cursor:])
    if h_span is None or t_span is None:
        return None
    return "".join(out), h_span, t_span


def call_gemini(client, prompt, max_retries=5):
    """Llama a Gemini con reintento + backoff -- la capa gratuita tiene cuota diaria baja."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4096,
        temperature=0.7,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config,
            )
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  aviso: fallo tras {max_retries} intentos ({e})")
                return None
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
    return None


def parse_batch_response(raw_text, n_expected):
    """Extrae el array JSON de la respuesta -- tolera un posible envoltorio ```json."""
    if raw_text is None:
        return []
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    if len(arr) != n_expected:
        print(f"  aviso: se esperaban {n_expected} elementos, llegaron {len(arr)} -- "
              f"se empareja por indice hasta el minimo")
    return arr


def augment_batch(client, batch_items):
    """batch_items: lista de instancias fuente (una por variante a generar,
    puede repetir la misma instancia si necesita >1 variante). Devuelve la
    lista de instancias nuevas que pasaron el filtro de calidad."""
    prompt_items = []
    for idx, inst in enumerate(batch_items, 1):
        head, tail = inst["h"]["name"], inst["t"]["name"]
        prompt_items.append(ITEM_TEMPLATE.format(
            idx=idx, text=inst["text"], head=head, tail=tail, relation=inst["relation"],
        ))
    prompt = "\n\n".join(prompt_items)

    raw_text = call_gemini(client, prompt)
    arr = parse_batch_response(raw_text, len(batch_items))

    results = []
    for inst, raw_item in zip(batch_items, arr):
        if not isinstance(raw_item, str):
            continue
        head, tail = inst["h"]["name"], inst["t"]["name"]
        parsed = strip_markers(raw_item.strip())
        if parsed is None:
            continue
        plain, h_span, t_span = parsed
        if plain[h_span[0]:h_span[1]].strip().lower() != head.strip().lower():
            continue
        if plain[t_span[0]:t_span[1]].strip().lower() != tail.strip().lower():
            continue
        results.append({
            "text": plain,
            "h": {"name": head, "pos": list(h_span)},
            "t": {"name": tail, "pos": list(t_span)},
            "relation": inst["relation"],
            "doc_id": f"{inst['doc_id']}_llmaug",
            "head_span": inst["head_span"],
            "tail_span": inst["tail_span"],
            "head_type": inst["head_type"],
            "tail_type": inst["tail_type"],
        })
    return results


def expand_targets(targets, variants_by_rel):
    """Una entrada por CADA variante a generar (repite la instancia si
    necesita mas de una), para poder trocear en lotes de tamano fijo sin
    importar cuantas variantes tenga cada relacion."""
    flat = []
    for inst in targets:
        for _ in range(variants_by_rel[inst["relation"]]):
            flat.append(inst)
    return flat


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", nargs="+", required=True,
                     help="Relaciones a aumentar, p.ej. APPLIED_TO ALTERNATIVE_NAME")
    ap.add_argument("--variants-per-instance", type=int, nargs="+", required=True,
                     help="Un numero de variantes por relacion, en el mismo orden "
                          "que --relations (misma longitud que --relations).")
    ap.add_argument("--train-path", default="../data/english/eng_train.txt")
    ap.add_argument("--out-path", default="../data/english/eng_train_llmaug.txt")
    args = ap.parse_args()

    if len(args.variants_per_instance) != len(args.relations):
        raise ValueError("--variants-per-instance debe tener la misma longitud que --relations")
    variants_by_rel = dict(zip(args.relations, args.variants_per_instance))

    client = genai.Client()

    train_path = Path(args.train_path)
    train = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    targets = [t for t in train if t["relation"] in variants_by_rel]
    print(f"{len(targets)} instancias fuente en {args.relations}")
    for rel, n in variants_by_rel.items():
        print(f"  {rel}: {n} variante(s) por instancia")

    flat = expand_targets(targets, variants_by_rel)
    batches = list(chunked(flat, BATCH_SIZE))
    print(f"{len(flat)} variantes a generar -> {len(batches)} llamadas de "
          f"{BATCH_SIZE} instancias cada una (antes habria sido {len(flat)} llamadas)")

    written = 0
    kept_by_rel = Counter()
    out_path = Path(args.out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, batch_items in enumerate(batches):
            for new_inst in augment_batch(client, batch_items):
                f.write(json.dumps(new_inst, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                kept_by_rel[new_inst["relation"]] += 1
            print(f"  lote {i + 1}/{len(batches)} ({(i + 1) * BATCH_SIZE} variantes procesadas), "
                  f"{written} generadas hasta ahora", flush=True)
            time.sleep(2)  # entre lotes, no entre instancias -- ya son muchas menos llamadas

    print(f"\nTotal generado: {written}")
    for rel, n in kept_by_rel.most_common():
        print(f"  {rel}: {n}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()

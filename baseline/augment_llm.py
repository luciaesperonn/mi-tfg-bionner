"""Data augmentation de clases minoritarias con Gemini (capa gratuita).

Parafrasea instancias positivas ya existentes en train, manteniendo las dos
menciones de entidad EXACTAS (marcadas por el modelo con [E1]/[E2] para poder
recalcular sus offsets de caracter de forma determinista, sin fiarse de que
el LLM cuente caracteres) y la misma relacion. Las generaciones que no
reproducen ambas menciones tal cual se descartan -- es el filtro de calidad
frente al riesgo de ruido de etiqueta.

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

SYSTEM_PROMPT = """You are a biomedical relation-extraction data augmentation \
assistant. Given a sentence and two entity mentions with the relation that \
holds between them, rewrite the sentence with different wording while \
preserving the same fact.

Rules:
- Keep both entity mentions character-for-character identical -- do not \
translate, abbreviate, expand, or pluralize them differently.
- The rewritten sentence must still clearly support the same relation \
between the same two entities -- do not weaken it into a vague or \
unrelated statement.
- Wrap the head entity in [E1] and [/E1], and the tail entity in [E2] and \
[/E2], exactly once each, around the verbatim entity text.
- Output ONLY the rewritten sentence with the markers. No explanation, no \
quotes, no markdown, no preamble."""

USER_TEMPLATE = """Sentence: {text}
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
    """Llama a Gemini con reintento + backoff -- la capa gratuita tiene RPM bajo."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
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


def augment_instance(client, inst, n_variants):
    head, tail = inst["h"]["name"], inst["t"]["name"]
    results = []
    for _ in range(n_variants):
        time.sleep(2)  # respeta el RPM bajo de la capa gratuita
        prompt = USER_TEMPLATE.format(
            text=inst["text"], head=head, tail=tail, relation=inst["relation"],
        )
        raw_text = call_gemini(client, prompt)
        if raw_text is None:
            continue
        parsed = strip_markers(raw_text.strip())
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

    written = 0
    kept_by_rel = Counter()
    out_path = Path(args.out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, inst in enumerate(targets):
            n_variants = variants_by_rel[inst["relation"]]
            for new_inst in augment_instance(client, inst, n_variants):
                f.write(json.dumps(new_inst, ensure_ascii=False) + "\n")
                written += 1
                kept_by_rel[new_inst["relation"]] += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(targets)} procesadas, {written} generadas hasta ahora")

    print(f"\nTotal generado: {written}")
    for rel, n in kept_by_rel.most_common():
        print(f"  {rel}: {n}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()

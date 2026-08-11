# Ejemplos: datos reales vs. datos aumentados con LLM

Para: revisión con el tutor. Contexto completo del experimento en
`DATA-AUGMENTATION-Y-FOCAL-LOSS-INFORME-FINAL.md` (mismo directorio) —
ese informe incluye el diagnóstico de por qué el augmentation no ayudó.

## Mecanismo: parafraseo con entidades ancladas

El método usado (`baseline/augment_llm.py`) le pide al LLM (Gemini
`gemini-3.5-flash`) que reescriba una frase real del dataset manteniendo
las dos entidades **exactamente iguales**, marcadas con `[E1]/[E2]`. Si el
texto dentro de los marcadores no coincide letra por letra con la entidad
original, la generación se descarta. Esto evita que el LLM invente un
hecho biomédico nuevo: solo cambia la forma de decir un hecho que ya es
verdad en el dataset original.

---

## Ejemplo 1 — ALTERNATIVE_NAME (fármaco ↔ abreviatura)

**Real** (`eng_train.txt`, doc `25591652_en`):
> "...First was **topiramate** (**TPM**) (n=12), followed by valproates (VPA)..."

Par de entidad: `topiramate` / `TPM`

**Aumentado, variante 1** (`eng_train_llmaug.txt`):
> "The antiepileptic drug **topiramate**, commonly abbreviated as **TPM**, was associated with generic-substitution-induced seizure worsening in 12 cases."

**Aumentado, variante 2**:
> "Among the patients experiencing worsening symptoms, **topiramate** (also known as **TPM**) was the most frequent drug involved."

Mismo par de entidad (`topiramate`/`TPM`) en las tres frases — solo cambia
la sintaxis alrededor.

---

## Ejemplo 2 — APPLIED_TO (prueba ↔ órgano)

**Real** (`eng_train.txt`, doc `25823269_en`):
> "...There were no changes in NT-proBNP levels, telangiectasias and anti-centromere antibodies, and EchoCG and **lung** test results..."

Par de entidad: `lung test` / `lung`

**Aumentado, variante 1**:
> "A clinical **lung test** is used to evaluate pulmonary function and assess damage within the **lung** tissue of patients with systemic sclerosis."

**Aumentado, variante 2**:
> "To determine the extent of disease in the **lung**, patients underwent a comprehensive **lung test** along with other diagnostic evaluations."

---

## Ejemplo 3 — pareja de entidad NUEVA (no parafraseo, generación 5B)

En una segunda ronda se probó generar con parejas de entidad que **no
existían** en el dataset original (vocabulario médico estándar,
verificado antes de pedírselo al LLM, no inventado por él):

**Aumentado** (`eng_train_llmaug_newpairs.txt`):
> "Contrast-enhanced ultrasound of the **kidney** revealed a well-circumscribed cortical mass, which was subsequently confirmed as a primary **renal** cell carcinoma upon histological evaluation."

Par de entidad: `kidney` / `renal` — no aparece en ningún ejemplo real del
dataset.

---

## El resultado cuantitativo (lo que probablemente le interesa más)

Macro F1 en protocolo **ciego calibrado** (el más realista, comparable con
CodaBench), seed 42, sobre las dos relaciones que se intentaron arreglar:

| Técnica | Macro F1 global | ALTERNATIVE_NAME | APPLIED_TO |
|---|---|---|---|
| 3A — baseline, sin tocar nada | 0.4298 | 0.131 | 0.062 |
| 4B — focal loss (sin datos nuevos) | 0.4371 | **0.173** | 0.259 |
| 4C — class weights (sin datos nuevos) | **0.4439** | 0.127 | 0.162 |
| 5A — + augmentation (parafraseo, 318 ejemplos) | 0.4155 | 0.111 | 0.245 |
| 5B — + parejas nuevas (20 ejemplos) | 0.4308 | 0.056 | 0.198 |
| 5C — augmentation + focal loss combinados | 0.4219 | 0.076 | **0.259** |

**Lo que falla:** `ALTERNATIVE_NAME` empeora cada vez que se le añade algo
relacionado con datos sintéticos (0.131 → 0.111 → 0.056 → 0.076), mientras
que las dos técnicas que no generan ni un dato nuevo (focal loss, class
weights) mejoran ambas relaciones objetivo sin ese coste.

**Diagnóstico (matriz de confusión, `ALTERNATIVE_NAME`, 94 casos gold en
5A):** el modelo predice `no_relation` en el 70-75% de los casos reales,
tanto antes como después de aumentar los datos. Es decir, el problema no
es "el modelo no ha visto suficientes ejemplos del patrón" — es que el
modelo no se atreve a predecir la clase (problema de confianza/umbral).
189 parafraseos nuevos no movieron esa aguja porque siguen cubriendo las
mismas ~95 parejas de entidad de siempre; en blind el modelo se encuentra
parejas nunca vistas y necesita generalizar el patrón relacional, no
memorizar más formas de decir lo mismo sobre las mismas parejas.

Detalle completo (diseño, matrices de confusión completas, las 14
relaciones × 6 experimentos, limitaciones) en
`DATA-AUGMENTATION-Y-FOCAL-LOSS-INFORME-FINAL.md`.

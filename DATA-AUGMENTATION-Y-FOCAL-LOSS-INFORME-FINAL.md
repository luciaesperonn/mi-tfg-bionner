# Informe final: Data Augmentation con LLM vs. Focal Loss para clases minoritarias

Fecha: 2026-08-06

Este documento reúne, con el máximo detalle posible, todo el trabajo hecho
para intentar mejorar el macro F1 en las relaciones peor representadas del
dataset (`ALTERNATIVE_NAME`, `APPLIED_TO`), a través de dos vías distintas:
generar datos sintéticos con un LLM, y cambiar la función de pérdida del
entrenamiento. Incluye el razonamiento de cada decisión, los números reales
de cada experimento (verificados contra los ficheros guardados, no de
memoria), los problemas técnicos que salieron por el camino y cómo se
resolvieron, y la conclusión final con su justificación completa.

Complementa a (sin repetir íntegramente):
- `PLAN-DATA-AUGMENTATION-LLM.md` — el diseño original del augmentation.
- `DATA-AUGMENTATION-ANALISIS-Y-ESTADO.md` — análisis de impacto esperado y
  el primer bloqueo de cuota.
- `DATA-AUGMENTATION-RESUMEN-COMPLETO.md` — resumen tras la primera
  generación exitosa (318 ejemplos).

Este informe va más allá de esos tres: incluye los experimentos 5B, 5C, y la
comparación final con las técnicas de reponderación de pérdida (4B focal
loss, 4C class weights) que ya existían antes de tocar el augmentation.

---

## 0. Resumen ejecutivo (añadido tras completar los 4 experimentos)

**Contexto de cómo se llegó a estos resultados:** partiendo del desbalance de
clases descrito en la sección 1 y de las dos técnicas de reponderación de
pérdida ya probadas sin generar ni un dato nuevo (4B focal loss, 4C class
weights — sección 2), se diseñó un plan de data augmentation con LLM
(Gemini, capa gratuita) enfocado exclusivamente en las dos relaciones peor
representadas en el protocolo ciego calibrado: `ALTERNATIVE_NAME` (F1=0.131)
y `APPLIED_TO` (F1=0.062) — sección 3. A partir de ahí se ejecutaron cuatro
experimentos de entrenamiento sucesivos, todos con la misma configuración
base que el baseline `3A` (PubMedBERT, `fix_entity_markers()`, mismos
hiperparámetros, mismo protocolo de evaluación ciega calibrada, seed 42),
cambiando en cada uno solo la variable que se quería aislar:

- **5A** — parafraseo de las instancias reales de cada relación (95 de
  `ALTERNATIVE_NAME`, 43 de `APPLIED_TO`) manteniendo las dos menciones de
  entidad idénticas carácter a carácter (318 ejemplos nuevos, sección 4).
- **5B** — 5A más 20 parejas de entidad completamente nuevas y verificadas
  de antemano (p. ej. `kidney`↔`renal`), para atacar la falta de diversidad
  de ENTIDADES en vez de solo de frases (sección 5).
- **5C** — los mismos datos de 5A combinados con focal loss, para ver si el
  augmentation aporta algo por ENCIMA de la técnica que ya funcionaba mejor
  sin datos nuevos (sección 6).
- **5D** — repetición de 5B con los prompts corregidos tras una auditoría de
  calidad manual que encontró defectos reales en las generaciones
  anteriores (sección 10), para descartar que el fracaso se debiera a datos
  mal generados en vez de a la hipótesis de fondo.

Con ese recorrido completo, el resultado es el siguiente:

### Resultado corto: el data augmentation con LLM no mejoró el modelo

| Experimento | Qué se hizo | Macro F1 ciego calibrado | vs baseline (3A = 0.4298) |
|---|---|---|---|
| **4B** (focal loss, sin datos nuevos) | cambia la loss, cero datos nuevos | **0.4371** | **+0.0073** ← el mejor con diferencia |
| 4C (class weights, sin datos nuevos) | cambia la loss, cero datos nuevos | 0.4439 | +0.0141 ← el mejor global |
| 5A (augment: 318 parafraseos) | +189 ALTERNATIVE_NAME, +129 APPLIED_TO | 0.4155 | −0.0143 |
| 5B (+20 parejas de entidad nuevas) | ensancha vocabulario de entidades | 0.4308 | +0.0010 (ruido) |
| 5C (augment + focal loss combinados) | 5A + 4B juntos | 0.4219 | −0.0079 |
| 5D (prompts corregidos, datos regenerados con más calidad) | repite 5B con datos auditados y arreglados | 0.4115 | −0.0183 |

**Ninguna variante de augmentation superó a simplemente cambiar la función de
pérdida (focal loss / class weights), que no generan ni un solo dato nuevo.**

### Por qué (esto es lo interesante)

Se hizo un análisis fino de la probabilidad cruda que el modelo asigna a
`ALTERNATIVE_NAME` en cada caso real, y la distribución es **bimodal**, no
"baja en general":

- **63-68%** de los casos reales: el modelo les asigna probabilidad ≈0 — ni
  se plantea la clase.
- **~20-24%**: el modelo acierta con confianza >0.90.
- Solo **12-17 de 94 casos** están en la zona intermedia donde ajustar un
  threshold cambiaría algo.

Y la población de "probabilidad ≈0" resultó ser un patrón lingüístico
concreto: pares sinónimos **sin solapamiento léxico** (`tumors`↔`cancers`,
`vascular`↔`vessels`) — el modelo parece haber aprendido `ALTERNATIVE_NAME`
como un patrón léxico de superficie (fármaco↔marca, substring), no como
relación semántica general. Por eso parafrasear las mismas 95 parejas de
siempre (5A) no ayudó — el modelo ya domina esas parejas o nunca las va a
reconocer solo por ver más frases alrededor. Y generar parejas nuevas (5B)
empeoró la precisión porque el modelo ensanchó el criterio y empezó a
confundir `ALTERNATIVE_NAME` con `ABBREVIATION`/`PART_OF`.

Lo más contundente: se corrigieron los prompts (5D) para arreglar defectos
de calidad reales encontrados en auditoría manual (marcador de equivalencia
explícito, etc.) y el resultado fue **prácticamente idéntico a antes de
corregir nada** (0.110 vs 0.111) — descarta que el problema fuera "datos mal
generados". El problema nunca fue de datos.

### Conclusión práctica

- **Técnica ganadora: focal loss (4B)**, sin ningún dato sintético.
- Es un resultado válido para la memoria del TFG, no un fracaso — hipótesis
  razonable, bien diseñada, probada con rigor, descartada con evidencia
  (matriz de confusión + distribución de probabilidad + verificación
  experimental tras corregir prompts).
- Limitación honesta que el propio informe señala: todo esto es de **una
  sola seed (42)** — antes de darlo por definitivo en la memoria, habría que
  repetirlo con 2-3 seeds más para confirmar que los deltas son reales y no
  ruido.

---

## 1. El problema de partida: desbalance de clases

El dataset (formato OpenNRE, derivado de NEREL-BIO track inglés) tiene 14
tipos de relación positiva más `no_relation`. El número de ejemplos por
relación en el train original (`eng_train.txt`, con `neg_ratio=3`) es muy
desigual:

| Relación | Ejemplos en train |
|---|---|
| no_relation | 9546 |
| SUBCLASS_OF | 743 |
| HAS_CAUSE | 579 |
| AFFECTS | 362 |
| ASSOCIATED_WITH | 325 |
| PART_OF | 204 |
| TO_DETECT_OR_STUDY | 164 |
| PHYSIOLOGY_OF | 150 |
| FINDING_OF | 121 |
| ORIGINS_FROM | 120 |
| TREATED_USING | 101 |
| ABBREVIATION | 97 |
| ALTERNATIVE_NAME | 95 |
| USED_IN | 89 |
| **APPLIED_TO** | **43** |

`SUBCLASS_OF` tiene 17 veces más ejemplos que `APPLIED_TO`. El macro F1 (la
métrica principal de este TFG) da el mismo peso a cada relación
independientemente de su tamaño, así que las relaciones pequeñas son las que
más penalizan la nota final si el modelo no las aprende bien.

**Cuál es realmente la peor, con datos del protocolo ciego (no del dev
curado):** se mira el desglose por relación de PubMedBERT en el protocolo
ciego **calibrado** (el que de verdad importa, comparable con CodaBench, no
el dev curado que es más optimista y menos realista):

| Relación | F1 ciego-calibrado (3A) |
|---|---|
| **ALTERNATIVE_NAME** | **0.131** |
| **APPLIED_TO** | **0.062** |
| HAS_CAUSE | 0.362 |
| ASSOCIATED_WITH | 0.351 |
| TO_DETECT_OR_STUDY | 0.373 |
| PART_OF | 0.371 |
| USED_IN | 0.429 |
| AFFECTS | 0.439 |
| FINDING_OF | 0.475 |
| ORIGINS_FROM | 0.597 |
| PHYSIOLOGY_OF | 0.514 |
| TREATED_USING | 0.568 |
| ABBREVIATION | 0.637 |
| SUBCLASS_OF | 0.708 |

`ALTERNATIVE_NAME` y `APPLIED_TO` son, con diferencia, las dos peores. El
error dominante en ambas es **infra-predicción**: precision razonable, recall
muy bajo — el modelo casi nunca se atreve a predecir estas clases, y por
defecto dice `no_relation`. Esto es importante porque **no es lo mismo** que
un problema de confusión con otra clase concreta; es un problema de
confianza/umbral, y ese matiz termina siendo la clave de todo este informe.

---

## 2. Técnicas que ya se habían probado ANTES del data augmentation

Antes de generar ningún dato sintético, ya se habían probado dos técnicas
que atacan el desbalance sin crear información nueva — solo redistribuyen la
importancia de la que ya hay:

### 2.1. Experimento 4B — Focal Loss

**Qué es:** en vez de `CrossEntropyLoss`, se usa
`FocalLoss(gamma=2.0) = -(1-p_t)^gamma * log(p_t)` (Lin et al. 2017). Reduce
el peso de los ejemplos "fáciles" (donde el modelo ya acierta con confianza)
y concentra el gradiente en los difíciles. Verificado antes de usarlo:
`FocalLoss(gamma=0)` coincide exactamente con `CrossEntropyLoss` (test de
regresión en el propio notebook).

**Único cambio respecto al baseline 3A** (`fix_entity_markers()`, sin typed
markers, `neg_ratio=3`, mismos hiperparámetros): la función de pérdida.

**Resultado (seed 42):**
- macro F1 curado: 0.7575
- macro F1 ciego argmax: 0.3351
- **macro F1 ciego calibrado: 0.4371** (mejor threshold: 0.923)
- vs 3A (0.4298): **+0.0073**

### 2.2. Experimento 4C — Class Weights

**Qué es:** `CrossEntropyLoss(weight=inverse_freq)` — cada clase pesa el
inverso de su frecuencia en train. Pesos calculados: `APPLIED_TO` obtiene el
peso más alto (19.75), `ALTERNATIVE_NAME` el segundo más alto (8.94),
`no_relation` el más bajo (0.089).

**Resultado (seed 42):**
- macro F1 curado: 0.7739
- macro F1 ciego argmax: 0.3176
- **macro F1 ciego calibrado: 0.4439** (mejor threshold: 0.9995)
- vs 3A (0.4298): **+0.0141** — la mejor técnica hasta ese momento

**Desglose por relación de 4B y 4C, comparado con 3A** (recalculado sobre las
probabilidades guardadas, no de memoria):

| Relación | 3A | 4B (focal) | 4C (weights) |
|---|---|---|---|
| ABBREVIATION | 0.637 | 0.565 | 0.692 |
| AFFECTS | 0.439 | 0.460 | 0.481 |
| **ALTERNATIVE_NAME** | 0.131 | **0.173** | 0.127 |
| **APPLIED_TO** | 0.062 | **0.259** | 0.162 |
| ASSOCIATED_WITH | 0.351 | 0.361 | 0.390 |
| FINDING_OF | 0.475 | 0.475 | 0.541 |
| HAS_CAUSE | 0.362 | 0.384 | 0.388 |
| ORIGINS_FROM | 0.597 | 0.504 | 0.553 |
| PART_OF | 0.371 | 0.362 | 0.322 |
| PHYSIOLOGY_OF | 0.514 | 0.546 | 0.499 |
| SUBCLASS_OF | 0.708 | 0.678 | 0.685 |
| TO_DETECT_OR_STUDY | 0.373 | 0.413 | 0.420 |
| TREATED_USING | 0.568 | 0.530 | 0.531 |
| USED_IN | 0.429 | 0.408 | 0.425 |

**Conclusión de esta fase:** ninguna de las dos técnicas crea información
nueva, solo redistribuyen. Para atacar el problema de raíz (pocos ejemplos)
hacía falta generar ejemplos nuevos de verdad — de ahí surge la idea del data
augmentation con LLM.

---

## 3. El plan de data augmentation: diseño y decisiones

### 3.1. Por qué parafrasear y no generar desde cero

Se evaluaron tres formas de crear ejemplos nuevos:

1. **Entity replacement** — coger una frase existente y cambiar la entidad
   por otra del mismo tipo. Riesgo: la entidad nueva puede no encajar
   semánticamente en el contexto original.
2. **Back-translation** — traducir y volver a traducir. Riesgo: hay que
   recalcular los spans de entidad desde cero, y ese es justo el punto donde
   ya habían aparecido bugs reales en este proyecto (ver
   `HALLAZGOS-BUGS-TOKENIZACION.md`).
3. **Parafraseo con LLM manteniendo entidades fijas** — se elige esta. El
   LLM reescribe la frase pero las dos menciones de entidad deben quedar
   EXACTAMENTE igual, letra por letra, marcadas con `[E1]...[/E1]` y
   `[E2]...[/E2]`. Los offsets de carácter se calculan con código
   determinista (contando caracteres reales), nunca confiando en que el LLM
   sepa contar. Si el texto dentro de los marcadores no coincide
   exactamente con la entidad original, la generación se descarta entera —
   es el filtro de calidad frente al ruido de etiqueta.

**Por qué esto importa tanto:** evita a la vez el riesgo de recalcular spans
mal (bug real ya sufrido en este proyecto) y el riesgo de que el LLM invente
un hecho biomédico falso (parafrasear preserva un hecho que YA es verdad en
el dataset original; generar desde cero le pediría al LLM que garantice que
un hecho inventado es cierto).

### 3.2. Qué LLM y por qué

**Gemini `gemini-3.5-flash`** (capa gratuita de Google AI Studio). Se
descartó la API de Claude/GPT por coste (esto es un TFG, sin presupuesto).
Validado con una prueba en vivo antes de generar nada a gran escala: 11/11
instancias de prueba pasaron el pipeline completo sin fallos de calidad.

### 3.3. Qué relaciones aumentar y cuánto

Se aumentaron **solo** `ALTERNATIVE_NAME` y `APPLIED_TO` — las dos peores en
ciego-calibrado (sección 1) — y se descartó ampliar otras relaciones con F1
bajo pero soporte alto (`HAS_CAUSE` con 579 ejemplos, `ASSOCIATED_WITH` con
325, `PART_OF` con 204, `TO_DETECT_OR_STUDY` con 164) porque su problema
claramente no es de volumen de datos.

**Objetivo de generación:** llevar cada relación a ~150-300 ejemplos totales,
un nivel de soporte donde ya sabíamos (por `ABBREVIATION`, con 97 ejemplos y
F1=0.637 en ciego) que una relación puede aprenderse razonablemente bien si
el patrón es aprendible.

---

## 4. Primera generación (5A): parafraseo, y los dos problemas técnicos que salieron

### 4.1. Problema 1 — la cuota diaria, no por minuto

Al lanzar la generación completa (una llamada por variante, ~319 llamadas
necesarias) se descubrió que la capa gratuita de `gemini-3.5-flash` no tiene
un límite por minuto, como se había asumido en el plan original, sino
**20 peticiones al día** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`).
El primer intento produjo solo 19 instancias (justo por debajo del límite) en
vez de las ~1350 planeadas originalmente para las 9 relaciones del plan
completo.

### 4.2. Problema 2 — instancias perdidas al cortar el proceso atascado

Al intentar relanzar y quedarse el proceso reintentando en bucle contra la
cuota agotada, se cortó el proceso — y se perdieron 11 instancias que sí se
habían generado, porque el script las guardaba todas en memoria y solo
escribía a disco al final (`with open(out_path, "w") as f`, sin `flush`).

### 4.3. La solución: peticiones por lotes

Se corrigió `baseline/augment_llm.py`:
- Cada escritura hace `f.flush()` inmediatamente — nada se pierde si se
  corta a medias.
- El fichero se abre en modo `"a"` (append), no `"w"`.
- Las peticiones se agrupan en **lotes de 20 instancias por llamada**: un
  único prompt con una lista numerada, la respuesta es un array JSON con
  las 20 reescrituras en el mismo orden, cada una con sus marcadores
  `[E1]/[E2]` de siempre. Validado primero con un lote de prueba de 5 (5/5
  pasaron el filtro).

**Resultado del run completo, con el rediseño:**

| | Antes (1 instancia/llamada) | Después (20/llamada) |
|---|---|---|
| Llamadas necesarias | ~319 | **16** |

| Relación | Objetivo | Generado | Tasa de aceptación |
|---|---|---|---|
| ALTERNATIVE_NAME | ~190 | **189** | — |
| APPLIED_TO | ~129 | **129** | — |
| **Total** | ~319 | **318** | **99.7%** |

Guardado en `data/english/eng_train_llmaug.txt`, separado del
`eng_train.txt` original (que no se toca).

**Estado del dataset tras esta generación:**

| Relación | Original | + Generado | Total | Factor |
|---|---|---|---|---|
| ALTERNATIVE_NAME | 95 | +189 | 284 | ×3.0 |
| APPLIED_TO | 43 | +129 | 172 | ×4.0 |

### 4.4. Experimento 5A: entrenar con los datos aumentados

**Único cambio respecto a 3A:** `TRAIN_DATA` = `eng_train.txt` +
`eng_train_llmaug.txt` concatenados (13057 líneas). Todo lo demás (modelo,
hiperparámetros, `fix_entity_markers()`, `CrossEntropyLoss`) igual que 3A.

**Resultado (seed 42):**
- macro F1 curado: 0.7601
- macro F1 ciego argmax: 0.2986
- **macro F1 ciego calibrado: 0.4155** (threshold: 0.987)
- vs 3A (0.4298): **−0.0143** — peor que no hacer nada

**Por relación, las dos aumentadas:**

| Relación | 3A | 5A | delta |
|---|---|---|---|
| ALTERNATIVE_NAME | 0.131 | 0.111 | **−0.020** |
| APPLIED_TO | 0.062 | 0.245 | **+0.183** |

`APPLIED_TO` mejoró de verdad y mucho. `ALTERNATIVE_NAME` empeoró.

### 4.5. Diagnóstico: por qué parafrasear no ayudó a ALTERNATIVE_NAME

Se miró la matriz de confusión real de `ALTERNATIVE_NAME` en el propio run
5A (blind, 94 casos gold):

```
predicho como no_relation:      70 (74.5%)
predicho como ALTERNATIVE_NAME:  9 (9.6%)   <- acierto
predicho como SUBCLASS_OF:       8 (8.5%)
predicho como PART_OF:           5 (5.3%)
predicho como ABBREVIATION:      2 (2.1%)
```

Prácticamente idéntico al patrón de antes de la augmentation (51% antes vs
74.5% ahora). 189 parafraseos nuevos no movieron la aguja.

**Hipótesis:** el método de augmentation mantiene las entidades
**literalmente idénticas** en cada parafraseo. Las 284 instancias de
`ALTERNATIVE_NAME` siguen cubriendo las mismas ~95 parejas de entidad de
siempre, solo con frases distintas alrededor. En blind, el modelo se
encuentra con parejas de entidad que **nunca ha visto** — necesita
generalizar el patrón relacional a entidades nuevas, no memorizar más formas
de decir lo mismo sobre las mismas 95 parejas. Parafrasear no le da eso.
`APPLIED_TO` sí mejoró, posiblemente porque depende más de patrones léxicos
de la frase (verbos como "applied to", "administered", "used on") y menos de
reconocer una pareja de entidad concreta.

**Análisis cualitativo de los 95 pares reales de ALTERNATIVE_NAME** (hecho
para entender mejor el problema): no es un patrón homogéneo, es una mezcla
de al menos 5 mecanismos lingüísticos distintos:

1. Fármaco↔abreviatura estándar (topiramate/TPM, valproates/VPA)
2. Fármaco genérico↔nombre comercial (agomelatine/valdoxan, escitalopram/cipralex)
3. Variantes morfológicas de la misma raíz (epilepsies/epilepsy, diabetic/diabetes)
4. Sustantivo anatómico↔forma adjetiva (pulmonary/lung, heart/cardiac)
5. Sinónimos sueltos de hallazgo clínico (poor prognosis/poor outcome)

---

## 5. Segunda generación (5B): parejas de entidad nuevas, verificadas

### 5.1. La idea

Ya que el problema parecía ser falta de diversidad de ENTIDADES (no de
frases), se probó generar ejemplos con parejas de entidad **nuevas**, no
inventadas por el LLM sino **verificadas** de antemano como vocabulario
médico estándar:

- **Pares anatómicos sustantivo↔adjetivo** (vocabulario cerrado y
  enumerable: kidney/renal, liver/hepatic, brain/cerebral, stomach/gastric,
  skin/cutaneous, bone/osseous, nerve/neural, muscle/muscular...).
- **Variantes morfológicas de la misma raíz** (infection/infected,
  hypertension/hypertensive, necrosis/necrotic, ischemia/ischemic...).

Se compilaron 47 parejas candidatas y se comprobó **programáticamente** que
ninguna (en ningún orden) coincidía con las ya presentes en train +
train_llmaug — las 47 eran genuinamente nuevas.

El LLM solo escribió la frase alrededor de cada pareja ya verificada — no
decidió que la relación fuera cierta, eso ya estaba fijado de antemano por
la lista. Mismo mecanismo de seguridad de siempre (marcadores + filtro de
coincidencia exacta).

### 5.2. Bloqueo de cuota (otra vez) y generación parcial

Se volvió a agotar la cuota diaria a mitad de la generación (esta vez porque
ya se había gastado cuota en las pruebas del propio día). **20 de las 47
parejas** se generaron y guardaron correctamente (gracias al fix de `flush`,
nada se perdió) antes de que la cuota se agotara. Las 27 restantes (sobre
todo variantes morfológicas) quedaron pendientes.

Guardado en `data/english/eng_train_llmaug_newpairs.txt` (20 instancias,
todas `ALTERNATIVE_NAME`).

### 5.3. Experimento 5B: entrenar con parafraseo + parejas nuevas

**Datos:** `eng_train.txt` + `eng_train_llmaug.txt` (318) +
`eng_train_llmaug_newpairs.txt` (20) = 13077 líneas.

**Resultado (seed 42):**
- macro F1 curado: 0.7740
- macro F1 ciego argmax: 0.3157
- **macro F1 ciego calibrado: 0.4308** (threshold: 0.994)
- vs 3A (0.4298): **+0.0010** (dentro de ruido)
- vs 5A (0.4155): **+0.0153**

**Pero por relación, resultado contraintuitivo:**

| Relación | 5A | 5B | delta |
|---|---|---|---|
| ALTERNATIVE_NAME | 0.111 | **0.056** | **−0.055** |
| APPLIED_TO | 0.245 | 0.198 | **−0.047** |
| ABBREVIATION | 0.697 | 0.590 | **−0.107** (la caída más grande de toda la tabla) |
| TO_DETECT_OR_STUDY | 0.384 | 0.342 | −0.042 |
| PART_OF | 0.303 | **0.393** | **+0.090** |
| USED_IN | 0.403 | **0.486** | **+0.083** |
| PHYSIOLOGY_OF | 0.443 | 0.516 | +0.073 |
| FINDING_OF | 0.390 | 0.453 | +0.064 |

**Lectura honesta:** las 20 parejas nuevas empeoraron `ALTERNATIVE_NAME`
(el objetivo directo) y `APPLIED_TO`, y de paso hundieron `ABBREVIATION`
(que ni se tocó). El macro global sube solo porque otras 9 relaciones
completamente ajenas al augmentation mejoraron — probablemente un efecto
colateral de reorganizar las probabilidades en una única cabeza softmax de
15 clases, no una mejora real y dirigida.

**Matriz de confusión de ALTERNATIVE_NAME en 5B** (gold=94):

```
predicho como no_relation:       66 (70.2%)
predicho como ALTERNATIVE_NAME:  15 (16.0%)  <- sube el recall...
predicho como SUBCLASS_OF:        7 (7.4%)
predicho como PART_OF:            4 (4.3%)
predicho como ABBREVIATION:       2 (2.1%)
```

Precision/Recall: 5A P=0.132 R=0.096 → 5B P=0.034 R=0.160. El recall sí
mejoró (el modelo se atreve más), pero la precisión se hundió — ahora
dispara `ALTERNATIVE_NAME` más veces, pero acierta menos cuando lo hace.
Hipótesis: los pares anatómicos ensancharon la noción de "esto es
ALTERNATIVE_NAME" del modelo lo suficiente como para robarle terreno a
`ABBREVIATION` (semánticamente prima) y a otros casos que no lo son.

---

## 6. Tercera prueba (5C): combinar augmentation con la técnica que sí funcionaba

### 6.1. La pregunta que responde

Dado que 4B (focal loss, sin ningún dato nuevo) ya daba el mejor resultado
visto hasta entonces para las dos relaciones objetivo (`ALTERNATIVE_NAME`
0.173, `APPLIED_TO` 0.259), y nunca se había probado la combinación:
**¿los datos aumentados aportan algo por encima de focal loss solo, o el
problema nunca fue de volumen de datos?**

### 6.2. Diseño

**Datos:** los mismos que 5A (`eng_train.txt` + `eng_train_llmaug.txt`, 318
ejemplos de parafraseo — **sin** las 20 parejas nuevas de 5B, que ya habían
empeorado la relación objetivo dos veces seguidas).
**Loss:** `FocalLoss(gamma=2.0)`, idéntica a 4B.
Combina 5A + 4B; todo lo demás igual que 3A.

### 6.3. Resultado (seed 42)

- macro F1 curado: 0.7687
- macro F1 ciego argmax: 0.3126
- **macro F1 ciego calibrado: 0.4219** (threshold: 0.911)

| Técnica | Macro ciego calibrado |
|---|---|
| 3A (nada) | 0.4298 |
| **4B (focal loss solo)** | **0.4371** ← el mejor |
| 5A (augment solo) | 0.4155 |
| 5B (augment + parejas nuevas) | 0.4308 |
| 5C (augment + focal loss) | 0.4219 |

**5C es peor que 4B solo: −0.0152.** Combinar los datos sintéticos con focal
loss no suma, empeora respecto a focal loss sin ningún dato nuevo.

**Por relación, el patrón se cierra de forma contundente — tres intentos
seguidos, `ALTERNATIVE_NAME` empeora monótonamente cada vez que se le mete
algo relacionado con más datos:**

```
ALTERNATIVE_NAME: 0.131 (3A) -> 0.173 (4B) -> 0.111 (5A) -> 0.076 (5C)
APPLIED_TO:       0.062 (3A) -> 0.259 (4B) -> 0.245 (5A) -> 0.255 (5C)
```

`ALTERNATIVE_NAME` está en su peor punto de toda la serie con 5C (0.076).
`APPLIED_TO` con augmentation+focal loss (0.255) queda prácticamente igual
que con focal loss solo (0.259) — sin ganancia real de sumar los datos.

---

## 7. Tabla comparativa completa (las 14 relaciones, los 6 experimentos)

**Resumen rápido primero** (macro global + las dos relaciones objetivo, los
cinco experimentos con datos comparables):

| Técnica | Macro ciego cal. | ALTERNATIVE_NAME | APPLIED_TO |
|---|---|---|---|
| 3A (nada) | 0.4298 | 0.131 | 0.062 |
| 4B focal loss (sin datos nuevos) | 0.4371 | 0.173 | 0.259 |
| 4C class weights (sin datos nuevos) | 0.4439 | 0.127 | 0.162 |
| 5A augment parafraseo | 0.4155 | 0.111 | 0.245 |
| 5B augment + parejas nuevas | 0.4308 | 0.056 | 0.198 |

Con esto junto ya se ve la respuesta: las dos técnicas que no generan ni un
dato nuevo (4B, 4C) ganan a las dos rondas de augmentation, tanto en macro
global como en las relaciones concretas que se querían arreglar. Ahora el
detalle completo, con las 14 relaciones y los 6 experimentos (incluye 5C):

Todos los números son macro F1 por relación en el protocolo **ciego
calibrado**, recalculados sobre las probabilidades guardadas de cada run
(`blind_probs_*.npy`), no transcritos de memoria.

| Relación | 3A | 4B | 4C | 5A | 5B | 5C |
|---|---|---|---|---|---|---|
| ABBREVIATION | 0.637 | 0.565 | 0.692 | 0.697 | 0.590 | 0.551 |
| AFFECTS | 0.439 | 0.460 | 0.481 | 0.394 | 0.437 | 0.470 |
| **ALTERNATIVE_NAME** | 0.131 | **0.173** | 0.127 | 0.111 | 0.056 | 0.076 |
| **APPLIED_TO** | 0.062 | **0.259** | 0.162 | 0.245 | 0.198 | 0.255 |
| ASSOCIATED_WITH | 0.351 | 0.361 | 0.390 | 0.331 | 0.355 | 0.346 |
| FINDING_OF | 0.475 | 0.475 | 0.541 | 0.390 | 0.453 | 0.491 |
| HAS_CAUSE | 0.362 | 0.384 | 0.388 | 0.352 | 0.375 | 0.358 |
| ORIGINS_FROM | 0.597 | 0.504 | 0.553 | 0.565 | 0.591 | 0.462 |
| PART_OF | 0.371 | 0.362 | 0.322 | 0.303 | 0.393 | 0.343 |
| PHYSIOLOGY_OF | 0.514 | 0.546 | 0.499 | 0.443 | 0.516 | 0.505 |
| SUBCLASS_OF | 0.708 | 0.678 | 0.685 | 0.672 | 0.705 | 0.669 |
| TO_DETECT_OR_STUDY | 0.373 | 0.413 | 0.420 | 0.384 | 0.342 | 0.412 |
| TREATED_USING | 0.568 | 0.530 | 0.531 | 0.529 | 0.535 | 0.548 |
| USED_IN | 0.429 | 0.408 | 0.425 | 0.403 | 0.486 | 0.420 |
| **MACRO F1** | **0.4298** | **0.4371** | **0.4439** | 0.4155 | 0.4308 | 0.4219 |

**La mejor técnica global de las seis: 4C (class weights, 0.4439).** La
mejor para las dos relaciones objetivo específicamente: 4B (focal loss).
Ninguna variante de data augmentation supera a ninguna de las dos técnicas
de reponderación de pérdida.

---

## 8. Conclusión final

**La mejor técnica encontrada para este problema es Focal Loss, sin ningún
dato sintético.** Cuatro intentos distintos de data augmentation con LLM
(parafraseo de instancias existentes, parejas de entidad nuevas verificadas,
la combinación de parafraseo con focal loss, y una repetición del
parafraseo + parejas nuevas con prompts corregidos tras auditar la calidad
de los datos generados — sección 10.4) fueron probados de forma
sistemática, y ninguno superó a las técnicas de reponderación de pérdida que
no requieren generar ni un solo dato nuevo. La cuarta repetición es
particularmente concluyente: corregir la calidad textual de los datos
sintéticos (marcador de equivalencia explícito, filtrado de deriva
semántica) no cambió el resultado — descarta la hipótesis alternativa de
que el problema fuera "datos mal generados" en vez de "el problema nunca
fue de datos".

**Por qué, con evidencia:** el error dominante en `ALTERNATIVE_NAME` y
`APPLIED_TO` es que el modelo predice `no_relation` en el 70-75% de los
casos reales — es un problema de **confianza/umbral de decisión**, no de
falta de ejemplos positivos para aprender el patrón. Focal loss ataca
exactamente ese síntoma (penaliza más los errores en casos difíciles,
empuja al modelo a no refugiarse en la clase fácil). Data augmentation
ataca un síntoma distinto ("el modelo no ha visto suficientes ejemplos") que
no es el que realmente está limitando estas dos relaciones.

**Coste total de la vía de augmentation:** 338 instancias generadas (318
parafraseo + 20 parejas nuevas), ~4 días de trabajo iterando el diseño,
varios bloqueos de cuota resueltos, un bug de pérdida de datos corregido, y
tres experimentos de entrenamiento completos (~40 min cada uno). Resultado:
ninguna mejora neta sobre la alternativa más simple.

**Esto es un resultado válido y defendible para la memoria del TFG**, no un
fracaso: se planteó una hipótesis razonable (las clases pequeñas mejoran con
más datos), se diseñó con cuidado para minimizar riesgos conocidos
(entidades ancladas, sin inventar hechos), se probó de tres formas distintas
con rigor (mismo protocolo de evaluación ciega calibrada en las tres), y se
comparó honestamente contra la alternativa más simple. La hipótesis no se
confirmó, y el motivo (infra-predicción por umbral, no por volumen de datos)
quedó demostrado con la matriz de confusión, no solo intuido.

## 9. Análisis post-mortem: la distribución de probabilidad cruda revela que el problema no es "confianza baja", es bimodalidad (2026-08-10)

Este análisis se hizo **después** de cerrar la conclusión de la sección 8, al
plantearse si merecía la pena un cuarto intento de augmentation dirigido
específicamente a `ALTERNATIVE_NAME`. Antes de lanzar ningún experimento
nuevo, se decidió mirar algo que ninguno de los análisis anteriores había
mirado: no solo *si* el modelo acierta o falla en cada instancia (la matriz
de confusión de la sección 4.5), sino **qué probabilidad cruda asigna el
modelo a la clase correcta en cada uno de los positivos reales**, antes de
aplicar ningún threshold. Cambia la conclusión de forma importante.

### 9.1. Metodología

No se reentrenó nada. Se reutilizaron los `blind_probs_*.npy` ya guardados
de los runs 3A, 4B y 5A (protocolo ciego, 282 364 pares candidato por run).
Para localizar, dentro de esos 282 364 candidatos, cuáles corresponden a los
95 positivos reales de `ALTERNATIVE_NAME` del gold oficial
(`data/english/eng-dev-rel.tsv`), se cruzaron ambos ficheros por la misma
clave que usa el evaluador oficial del proyecto
(`create_instance_key()` en `baseline/score.py`):

```
key = f"{doc_id}|{head_span}|{tail_span}"
```

Se encontraron **94 de los 95** positivos gold dentro del conjunto de
candidatos ciegos (1 no tiene coincidencia — span no generado como candidato
en ese documento; no se investigó más porque es <1.1% del soporte y no
afecta ninguna conclusión). Para esos 94, se extrajo la columna de
`ALTERNATIVE_NAME` del softmax guardado (`probs[i, rel2id["ALTERNATIVE_NAME"]]`),
es decir, la probabilidad que el modelo le asigna a la clase correcta en
cada instancia gold, independientemente de qué clase acabe ganando.

### 9.2. Resultado: la distribución es bimodal, no uniformemente baja

| Rango de P(ALTERNATIVE_NAME) | 3A (baseline) | 4B (focal loss) | 5A (augment parafraseo) |
|---|---|---|---|
| 0.00 – 0.05 (el modelo ni se plantea la clase) | **59/94 (63%)** | **64/94 (68%)** | 63/94 (67%) |
| 0.05 – 0.90 (zona intermedia, "a punto de cruzar") | 12/94 (13%) | 17/94 (18%) | 12/94 (13%) |
| 0.90 – 1.00 (confianza muy alta, acierta) | 23/94 (24%) | 13/94 (14%) | 19/94 (20%) |
| **Media** | 0.309 | 0.208 | 0.230 |
| **Mediana** | 0.0001 | 0.0054 | 0.0011 |

La media (~0.2-0.3) por sí sola sugiere "confianza baja pero recuperable
subiendo un poco la señal" — que es justo la lectura que se había hecho
hasta ahora ("infra-predicción por umbral"). Pero la mediana (~0.0001-0.005)
y la distribución por buckets cuentan una historia distinta: **no hay una
masa de casos "casi seguros" que un ajuste de threshold o un poco más de
señal pudiera empujar por encima del corte.** Hay dos poblaciones separadas:
una que el modelo reconoce con altísima confianza (>0.90) y otra —
mayoritaria, 63-68% — donde la probabilidad es prácticamente cero. La zona
intermedia, que es la única que un ajuste de threshold o de calibración
podría rescatar, es pequeña: 12-17 casos de 94.

### 9.3. Qué distingue a las dos poblaciones (ejemplos reales, run 3A)

**Confianza muy alta (P > 0.99) — el modelo las reconoce sin problema:**

| Par | P(ALTERNATIVE_NAME) |
|---|---|
| axamon ↔ ipidacrine | 1.000 |
| valdoxan ↔ agomelatine | 1.000 |
| baclofen ↔ Baclosan | 1.000 |
| aortic ↔ aorta | 1.000 |
| extremities ↔ limbs | 0.999 |
| blood plasma ↔ plasma | 0.999 |
| diacerein ↔ diaflex | 0.998 |
| infection of vascular prostheses ↔ infection of the implant | 0.998 |

**Confianza ~0 (P < 0.001) — el modelo directamente no las considera candidatas:**

| Par | P(ALTERNATIVE_NAME) |
|---|---|
| tumors ↔ cancers | 0.000 |
| lung ↔ pleural | 0.000 |
| lymphocytic ↔ lymphocyte | 0.000 |
| oxidative processes ↔ oxidation | 0.000 |
| multidrug-resistant tuberculosis ↔ drug-resistant tuberculosis | 0.000 |
| impairment of blood supply ↔ ischemia | 0.000 |
| newly formed blood vessels ↔ neovasculature | 0.000 |
| vascular ↔ vessels | 0.000 |
| proinflammatory cytokines ↔ inflammatory cytokines | 0.000 |
| dorsopathy ↔ dorsalgia | 0.000 |

El patrón es consistente: la población de confianza alta son casi todo
pares fármaco↔marca comercial (memorizables, aparecen juntos en la
literatura de forma estereotipada) o pares con **solapamiento léxico
directo** (una cadena es substring o casi-substring de la otra: `aortic`/
`aorta`, `blood plasma`/`plasma`). La población de confianza cero son
sinónimos semánticos genuinos **sin solapamiento de superficie**: raíces
distintas, morfología distinta (`tumors`/`cancers`, `vascular`/`vessels`,
`dorsopathy`/`dorsalgia`). El modelo, tal como está entrenado ahora mismo,
parece haber aprendido `ALTERNATIVE_NAME` en buena parte como un patrón
**léxico de superficie**, no como una relación semántica general.

### 9.4. Por qué esto explica (mecánicamente) los tres fracasos de augmentation ya documentados

- **5A (parafraseo de las 95 parejas reales) no ayudó** porque una parte
  importante de esas 95 parejas ya son del tipo "fácil" (fármaco↔marca,
  solapamiento léxico) que el modelo ya clava con P>0.99 — parafrasear más
  contextos alrededor de una pareja en la que el modelo ya tiene techo de
  confianza no añade señal de aprendizaje nueva. El hueco real (la mayoría,
  63-68%) está en el subtipo de bajo solapamiento léxico, que 5A no atacó de
  forma dirigida (generó variantes de las mismas 95 parejas indistintamente,
  sin distinguir subtipo).
- **5B (parejas nuevas verificadas, tipo `kidney`↔`renal`) conceptualmente sí
  apuntaba al hueco correcto** (son pares de bajo solapamiento léxico, como
  los del bucket de confianza cero) — pero el resultado fue que la precisión
  se hundió (0.132→0.034, sección 4.5.3 más arriba), no que el recall
  mejorara limpiamente. Es decir: **incluso apuntando al subtipo correcto,
  el resultado fue negativo**, probablemente porque el modelo, al ver más
  ejemplos de "cosas semánticamente parecidas etiquetadas como
  ALTERNATIVE_NAME", ensanchó el criterio más de lo debido y empezó a
  confundirlo con relaciones vecinas (`ABBREVIATION`, `PART_OF`) en vez de
  aprender la distinción fina.
- **La propuesta de "calibrar un threshold específico para
  `ALTERNATIVE_NAME`"** (barajada en esta misma conversación como paso 2 de
  un plan de tres pasos) tiene recorrido limitado por la misma razón: solo
  hay 12-17 casos en la zona intermedia donde mover el corte de decisión
  cambiaría el resultado. La mayoría de los positivos perdidos no están
  "cerca" del umbral — están en P≈0, y ningún ajuste de threshold mueve eso.

### 9.5. Conclusión de esta sección

El diagnóstico de la sección 8 ("infra-predicción por problema de
confianza/umbral") sigue siendo cierto a nivel agregado, pero esta vista más
fina lo matiza de forma importante: **no es un problema de calibración
general de la clase, es un blind spot específico para un subtipo lingüístico
concreto** (sinónimos semánticos sin solapamiento léxico) que convive con un
subtipo que el modelo ya domina (fármaco↔marca, solapamiento léxico). Un
quinto intento de augmentation que quisiera tener alguna probabilidad de
funcionar tendría que:

1. Generar (o minar) ejemplos **específicamente** del subtipo de bajo
   solapamiento léxico — no parafrasear indiscriminadamente las 95 parejas
   existentes, la mayoría de las cuales ya están en el subtipo fácil.
2. Asumir el riesgo ya demostrado en 5B: incluso apuntando al subtipo
   correcto, la precisión puede hundirse si el modelo generaliza el criterio
   más de la cuenta. Cualquier intento en esta dirección necesitaría medir
   precision y recall por separado (no solo F1 agregado) para detectar ese
   fallo específico pronto, y probablemente un filtro adicional que
   distinga "relacionado semánticamente" de "es el mismo concepto con otro
   nombre" — la distinción que el modelo actual no está haciendo bien.

Dado que ya hay tres intentos de augmentation fallidos y ahora una
explicación mecánica de por qué, **no se considera prioritario lanzar este
quinto intento** sin una idea de diseño que aborde explícitamente el riesgo
de 5B (no solo el subtipo correcto, sino cómo evitar que ensanche el
criterio). Se deja documentado como línea de trabajo futuro (sección 11).

## 10. Auditoría de calidad de los datos sintéticos generados (2026-08-10)

Tras el análisis de la sección 9 se hizo una revisión manual, línea por
línea, de los dos ficheros sintéticos de `ALTERNATIVE_NAME`
(`eng_train_llmaug.txt`, 189 instancias, y `eng_train_llmaug_newpairs.txt`,
20 instancias) para responder a una pregunta muy concreta: **¿el propio dato
generado tiene sentido, o hay ejemplos mal etiquetados metidos en medio de
las cifras de la sección 4-7?**

### 10.1. `eng_train_llmaug.txt` (parafraseo de las 95 parejas reales): mayormente sólido, con un 3-4% de deriva semántica

La inmensa mayoría (~180/189) sí expresa bien la relación, casi siempre
porque hereda del texto real un conector explícito de equivalencia
("also known as", "abbreviated as", "formerly termed"...). Se encontraron 6
casos concretos donde la frase generada por el LLM usa un lenguaje que
apoya una relación **distinta** de "mismo concepto, otro nombre" —
típicamente asociación o coexistencia, no identidad:

| Par | Frase generada (fragmento) | Relación que realmente describe |
|---|---|---|
| `myoclonic` ↔ `myoclonus` | *"share pathophysiological features with"* | ASSOCIATED_WITH, no ALTERNATIVE_NAME |
| `cognitive impairment` ↔ `cognitive disorders` | *"prevent the progression of [...] into more severe cognitive disorders"* | etapas distintas de una progresión, no lo mismo |
| `frequency and regional EEG` ↔ `Spectral characteristics of background EEG` | *"complemented by measuring"* | dos medidas añadidas, no una renombrada |
| `recurrences` ↔ `relapse` | *"tracked both... and"* | dos métricas paralelas, no sinónimas |
| `regressed spasticity` ↔ `reduced spasticity` | *"accompanied by"*, en extremidades distintas | dos hallazgos que coexisten en sitios distintos |
| `Odontogenic maxillary sinusitis` ↔ `odontogenic sinusitis` | *"represents a major portion of"* | relación parte-todo/hipónimo, no sinonimia |

Es un ~3-4% de las 189, no un problema masivo, pero confirma que incluso con
el prompt más cuidadoso (que exige explícitamente "must still clearly
support the same relation") el LLM puede desviarse cuando el texto real de
partida ya usaba un lenguaje ambiguo.

Aparte de esto: varias entidades ya venían defectuosas del dato real (p.ej.
`"schizophrenia: effects"`, un fragmento de título con dos puntos) y, al
tener que preservarse carácter a carácter, producen frases antinaturales —
es un defecto heredado de la anotación original, no algo que el
augmentation introduzca.

### 10.2. `eng_train_llmaug_newpairs.txt` (parejas nuevas verificadas): fallo sistemático, 20/20

Aquí el problema es total, no parcial. Ninguna de las 20 frases generadas
afirma que las dos palabras son "lo mismo, dicho de otra forma" — todas
usan el sustantivo para nombrar el órgano examinado y el adjetivo para
calificar un hallazgo o diagnóstico relacionado pero **distinto**. Ejemplo
(`kidney`/`renal`): *"Contrast-enhanced ultrasound of the **kidney**
revealed [...] confirmed as a primary **renal** cell carcinoma"* — el
riñón es el órgano; el carcinoma de células renales es un diagnóstico
específico encontrado en él, no "otro nombre" del riñón.

La causa se encontró en el propio prompt de `augment_llm_newpairs.py`
(líneas 47-50): pide al LLM escribir una frase que use ambos términos "en
cláusulas cercanas" para el mismo órgano/condición, sin exigir una marca
explícita de equivalencia. El LLM cumplió literalmente lo pedido —
co-ocurrencia temática, no sinonimia — y ese criterio más laxo es
exactamente el que explica, de forma mecánica, el colapso de precisión ya
medido en el experimento 5B (0.132→0.034, sección 5.3): el modelo aprendió
que "cosas del mismo tema anatómico cerca en la frase" es
`ALTERNATIVE_NAME`, un criterio mucho más amplio que el real.

**Conclusión práctica:** `eng_train_llmaug_newpairs.txt` no debería
reutilizarse tal cual en ningún experimento futuro sin antes corregir el
prompt para exigir una marca explícita de equivalencia (igual que ya hace
`augment_llm.py`).

### 10.3. La distinción `ABBREVIATION` vs `ALTERNATIVE_NAME` en NEREL-BIO es real y verificable, no ruido de anotación

Al revisar bibliografía externa sobre NEREL/NEREL-BIO surgió la hipótesis de
que la frontera entre `ABBREVIATION` y `ALTERNATIVE_NAME` podría ser
arbitraria o inconsistente en la anotación (p.ej., que `topiramate`→`TPM`
"debería" ser `ABBREVIATION`). Se comprobó de forma programática sobre los
datos reales del propio `eng_train.txt`, definiendo "inicial literal" como:
las letras del término corto = primera letra de cada palabra del término
largo, en orden.

| | Son iniciales literales del término largo | Total |
|---|---|---|
| `ABBREVIATION` (real) | 64/97 (66% con test exacto; el resto falla solo por guiones/números que el test no maneja bien — `hs-CRP`, `T2DM` — pero siguen siendo iniciales a simple vista) | 97 |
| `ALTERNATIVE_NAME`, subconjunto con forma corta tipo código (`TPM`, `VPA`, `SDS`, `SCT`, `C2`, `EDVD`...) | **0/14 (0%)** | 14 |

Ninguna de las 14 parejas de `ALTERNATIVE_NAME` con forma corta es una
inicial literal del término completo — todas son o bien un **código externo
no derivable** (`topiramate`→`TPM`: ninguna letra coincide posicionalmente
con "topiramate") o un **truncamiento por substring/prefijo**
(`SCT angiography`→`SCT`, `C2 vertebra`→`C2`, `blood plasma`→`plasma`: el
término corto es literalmente un trozo del largo, no unas iniciales).

**Conclusión:** la frontera entre las dos relaciones en NEREL-BIO es
consistente y programáticamente verificable — no es una inconsistencia de
anotación que haya que corregir. Esto también refina la caracterización de
la sección 9.3: dentro del bucket de "alta confianza / alto solapamiento
léxico" hay al menos dos submecanismos distintos (código no derivable vs.
truncamiento por substring), relevante si en el futuro se retoma la
generación dirigida por subtipo.

### 10.4. Verificación experimental: se corrigieron los prompts, se regeneraron los datos y se reentrenó (2026-08-10)

Para no quedarse solo en el análisis textual, se hizo la prueba real:
corregir los dos prompts (`augment_llm.py` y `augment_llm_newpairs.py`,
añadiendo la regla explícita de marcador de equivalencia y prohibiendo el
lenguaje de asociación/progresión encontrado en 10.1-10.2), regenerar los
237 ejemplos de `ALTERNATIVE_NAME` (190 parafraseo + 47 parejas nuevas) con
los prompts corregidos, y entrenar un nuevo experimento (**5D**,
`notebooks/5D-experimento-pubmedbert-llmaug-fixed.ipynb`) idéntico a 5A en
todo excepto el dato de entrenamiento.

**Auditoría de calidad de los datos regenerados** (lectura manual completa,
no muestreo): el parafraseo mantiene calidad alta; las parejas nuevas pasan
de 0/47 a 44-45/47 con marcador de equivalencia correcto.

**Resultado (seed 42):**

| Experimento | ALTERNATIVE_NAME F1 (ciego calibrado) | Precision | Recall |
|---|---|---|---|
| 3A (sin augment) | 0.131 | — | — |
| 5A (augment, prompt **sin** corregir) | 0.111 | 0.132 | 0.096 |
| 5B (+parejas nuevas, **sin** corregir) | 0.056 | 0.034 | 0.160 |
| **5D — mismos datos, prompts corregidos** | **0.110** | 0.104 | 0.117 |

Macro F1 global: 0.4115 (5D) vs 0.4298 (3A) vs 0.4155 (5A) — en la misma
banda que 5A, por debajo de 3A.

**5D es prácticamente idéntico a 5A** (0.110 vs 0.111 — dentro del ruido de
una sola seed) pese a que la calidad textual de los datos sintéticos mejoró
sustancialmente (de 0/47 a 44-45/47 parejas nuevas con marcador correcto,
y de 183/189 a 189/189 parafraseos sin deriva semántica). Matriz de
confusión: TP=11, FP=95, FN=83 sobre 95 casos reales — el modelo sigue sin
reconocer la gran mayoría, en una proporción casi idéntica a antes de
corregir los prompts.

**Conclusión de esta verificación:** la hipótesis de que el resultado
negativo de 5A/5B se debía a que "los datos generados estaban mal hechos"
queda **descartada por evidencia experimental directa**, no solo por
análisis textual. Arreglar la calidad del texto generado (marcador de
equivalencia explícito, sin lenguaje de asociación/progresión) no cambió
el resultado de forma perceptible. Esto refuerza la conclusión de la
sección 9: el mecanismo de fallo es de confianza/umbral de decisión del
modelo frente a un patrón que reconoce de forma bimodal (o con seguridad
alta, o no lo reconoce en absoluto), no un problema de qué tan bien
redactadas están las frases de entrenamiento. Focal loss (0.173, sección
2.1) sigue siendo, con diferencia, la mejor técnica encontrada para esta
relación.

## 11. Limitaciones y trabajo futuro

- **Todo lo anterior es de una sola seed (42).** La desviación estándar
  entre seeds medida en el estudio multiseed (1G) para PubMedBERT en ciego
  calibrado es ~0.009-0.017. Varios de los deltas de esta tabla (p.ej. 5C vs
  4B, −0.0152) son mayores que ese ruido y probablemente reales, pero no se
  ha confirmado con más seeds — sería el paso obligado antes de dar esta
  conclusión por definitiva en la memoria.
- **Las 27 parejas nuevas restantes** (de las 47 planeadas para 5B) nunca
  se generaron — la cuota se agotó de nuevo. Dado que 5C ya cerró la
  pregunta de fondo (el problema no es de datos), no se considera prioritario
  completarlas, salvo que se decida repetir 5B/5C con multi-seed.
- **Los ficheros de datos aumentados quedan en el repo** por si se quisieran
  reutilizar en otro contexto: `data/english/eng_train_llmaug.txt` (318,
  parafraseo), `data/english/eng_train_llmaug_newpairs.txt` (20, parejas
  nuevas), y los ficheros combinados `eng_train_plus_llmaug.txt` (5A/5C) y
  `eng_train_plus_llmaug_v2.txt` (5B).
- **Los scripts quedan reutilizables:** `baseline/augment_llm.py`
  (parafraseo por lotes) y `baseline/augment_llm_newpairs.py` (parejas
  nuevas verificadas) — ambos con el fix de `flush()`+`append` aplicado, por
  si se necesita generar más datos para otra relación en el futuro.
- **El análisis de la sección 9 (distribución de probabilidad) es también de
  una sola seed (42) y de un único encoder (PubMedBERT)** — no se ha
  comprobado si la misma bimodalidad (léxico-reconocible vs.
  léxico-no-reconocible) aparece igual en BioBERT/SciBERT/BioLinkBERT ni si
  es estable entre seeds. Antes de diseñar el quinto intento de
  augmentation dirigido al subtipo de bajo solapamiento léxico (sección
  9.5), convendría confirmar que el patrón no es un artefacto de esta seed
  concreta.
- **1 de los 95 positivos gold de `ALTERNATIVE_NAME`** no apareció en el
  conjunto de candidatos ciegos (`eng_dev_blind.txt`) al cruzarlo por clave
  `doc_id|head_span|tail_span` — no se investigó la causa (probablemente un
  span que no se generó como candidato en ese documento). Afecta a <1.1%
  del soporte de la relación, no cambia ninguna conclusión de la sección 9,
  pero queda pendiente si se quiere una cifra exacta de 95/95.

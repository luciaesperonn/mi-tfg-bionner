# Data Augmentation con LLM — resumen completo de todo el proceso

Fecha: 2026-08-06

Este documento cuenta la historia completa, de principio a fin: qué se decidió
hacer, por qué, qué se generó de verdad, qué problemas salieron por el camino
y qué se descubrió que cambia el siguiente paso. Complementa a los otros dos
documentos de esta carpeta:
- `PLAN-DATA-AUGMENTATION-LLM.md` — el diseño original, antes de ejecutar nada.
- `DATA-AUGMENTATION-ANALISIS-Y-ESTADO.md` — el análisis de impacto esperado y
  el primer intento de ejecución (bloqueado por cuota).

## 1. El problema de partida

El dataset tiene un desbalance de clases fuerte: `SUBCLASS_OF` tiene 743
ejemplos en train, `APPLIED_TO` solo 43 (17 veces menos). Macro F1 pesa igual
a cada relación, así que las clases pequeñas son las que más penalizan la
nota final. Ya se había intentado arreglar con `neg_ratio`, class weights y
focal loss (ver sección 7) — ninguna de ellas crea información nueva, solo
redistribuye la importancia de la que ya hay. Para eso hace falta generar
ejemplos nuevos de verdad: data augmentation.

## 2. La decisión de diseño: parafrasear, no generar desde cero

Se evaluaron tres formas de generar ejemplos nuevos (entity replacement,
back-translation, parafraseo con LLM manteniendo entidades fijas) y se eligió
la tercera **a propósito**, por dos motivos que siguen siendo válidos hoy:

- **Parafrasear preserva un hecho que ya es verdad** en el dataset original;
  generar desde cero le pide al LLM que invente un hecho biomédico nuevo y
  garantice que cumple la relación — mucho más fácil que alucine algo
  plausible pero falso.
- **Las entidades quedan ancladas con marcadores `[E1]/[E2]` inline**, así que
  los offsets de carácter se calculan de forma determinista con código
  normal, sin que el LLM tenga que contar caracteres ni recalcular nada. Esto
  evita justo el tipo de bug documentado en `HALLAZGOS-BUGS-TOKENIZACION.md`.

Modelo usado: **Gemini `gemini-3.5-flash`** (capa gratuita), validado con
pruebas en vivo (11/11 instancias de prueba pasaron el pipeline completo sin
fallos de calidad antes de lanzar nada a gran escala).

## 3. Qué relaciones se decidió aumentar, y por qué justo esas dos

Se analizó el desglose por relación en **ciego calibrado** (el protocolo
realista, comparable con CodaBench) de los 4 encoders del experimento
multi-seed (1G), no solo de un modelo suelto:

| Relación | F1 ciego-calibrado (media 4 encoders) | Soporte en train |
|---|---|---|
| **ALTERNATIVE_NAME** | **0.128** | 95 |
| **APPLIED_TO** | **0.132** | 43 |
| HAS_CAUSE | 0.359 | 579 |
| ASSOCIATED_WITH | 0.372 | 325 |
| TO_DETECT_OR_STUDY | 0.373 | 164 |
| PART_OF | 0.374 | 204 |
| USED_IN | 0.450 | 89 |
| ORIGINS_FROM | 0.505 | 120 |
| ... | ... | ... |
| SUBCLASS_OF | 0.691 | 743 |

`ALTERNATIVE_NAME` y `APPLIED_TO` son las dos peores **en los 4 encoders sin
excepción**, y su error dominante es infra-predicción (precision razonable,
recall muy bajo — la mayoría de los casos se predicen como `no_relation`,
firma clásica de "el modelo apenas ha visto ejemplos positivos"). Se
descartó ampliar las otras relaciones (`HAS_CAUSE`, `ASSOCIATED_WITH`,
`PART_OF`, `TO_DETECT_OR_STUDY`) a pesar de tener F1 bajo, precisamente
porque **ya tienen soporte de sobra** (`HAS_CAUSE` tiene más ejemplos que
casi cualquier otra relación y aun así rinde mal) — su problema no es de
volumen de datos, así que generar más no lo iba a arreglar (ver sección 7
para lo que sí ayuda ahí).

**Techo de mejora estimado:** llevando estas dos relaciones a un nivel medio
(~0.45 en ciego-calibrado) se estimó una ganancia de **+0.045 a +0.05 de
macro F1** — comparado con el ruido entre seeds medido (±0.01-0.02), sería
una mejora real, no un espejismo, aunque tampoco una transformación
dramática.

## 4. Ejecución: lo que pasó de verdad (con dos problemas por el camino)

**Primer intento — bloqueado.** Al lanzar la generación completa se
descubrió que la capa gratuita de `gemini-3.5-flash` no tiene límite por
minuto sino **20 peticiones al día** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`).
Con una llamada por variante (~320 necesarias), esto era inviable sin esperar
semanas o pagar. Además, al cortar el proceso atascado en reintentos, se
perdieron 11 instancias ya generadas porque el script las guardaba todas en
memoria y solo escribía a disco al final (`with open(..., "w") as f`, sin
`flush`).

**Rediseño — por lotes.** Se corrigieron ambos problemas en
`baseline/augment_llm.py`:
- Cada escritura hace `f.flush()` inmediatamente (nada se pierde si se corta
  a medias) y el fichero se abre en modo `"a"` (append), no `"w"`.
- Las peticiones se agrupan en **lotes de 20 instancias por llamada** (antes
  era 1 instancia = 1 llamada): un único prompt con una lista numerada de 20
  frases, la respuesta es un array JSON con las 20 reescrituras en el mismo
  orden, cada una con sus marcadores `[E1]/[E2]` de siempre. Se validó primero
  con una prueba en vivo de un lote de 5 (5/5 pasaron el filtro) antes de
  lanzar el run completo.

**Resultado del run completo (`ALTERNATIVE_NAME` + `APPLIED_TO`):**

| | Llamadas necesarias | Real |
|---|---|---|
| Antes del rediseño (1 instancia/llamada) | ~319 | — |
| Después (20 instancias/llamada) | — | **16 llamadas** |

| Relación | Objetivo | Generado (pasó el filtro) | Tasa de aceptación |
|---|---|---|---|
| ALTERNATIVE_NAME | ~190 | **189** | — |
| APPLIED_TO | ~129 | **129** | — |
| **Total** | ~319 | **318** | **99.7%** |

Las 16 llamadas caben cómodamente en la cuota diaria gratuita — el bloqueo
quedó resuelto sin necesidad de pedir una API de pago a nadie.

## 5. Estado actual del dataset aumentado

Guardado en `data/english/eng_train_llmaug.txt` (separado de `eng_train.txt`,
que queda intacto — el aumentado es un añadido opcional, se concatena solo
cuando se monte el experimento que lo use):

| Relación | Original | + Generado | Total | Factor |
|---|---|---|---|---|
| ALTERNATIVE_NAME | 95 | +189 | **284** | **×3.0** |
| APPLIED_TO | 43 | +129 | **172** | **×4.0** |

## 6. Lo que se decidió NO hacer (y por qué)

Se planteó ampliar la generación a una tabla mucho más grande — 8 relaciones,
~250-300 ejemplos cada una, ~2200-2300 en total, incluyendo `USED_IN`,
`ORIGINS_FROM`, `TO_DETECT_OR_STUDY`, `ASSOCIATED_WITH`, `PART_OF`,
`HAS_CAUSE`. Se rechazó porque, según los datos reales de la sección 3, la
mayoría de esas relaciones **no tienen un problema de volumen** — `HAS_CAUSE`
con 579 ejemplos ya es de las que más soporte tiene y sigue rindiendo mal;
`ORIGINS_FROM` (120 ejemplos) ya es de las relaciones que **mejor** rinden.
Generar el doble de datos sintéticos sobre clases que no lo necesitan solo
diluiría el dataset (un ~18% de datos sintéticos añadidos sin evidencia de
que ayuden) sin fundamento medido.

## 7. Hallazgo importante para el siguiente paso: otra palanca que sí ayuda a las relaciones "grandes pero malas"

Mientras se decidía qué hacer con esas relaciones descartadas, se revisó el
desglose por relación de dos experimentos que ya estaban corridos
(`4B-pubmedbert-focalloss`, `4C-pubmedbert-classweights`) contra el baseline
`3A-pubmedbert-fixed`, justo en las relaciones grandes-pero-malas:

| Relación | 3A (baseline) | 4B focal loss | 4C class weights |
|---|---|---|---|
| HAS_CAUSE | 0.362 | 0.384 (+0.022) | **0.388 (+0.026)** |
| ASSOCIATED_WITH | 0.351 | 0.361 (+0.010) | **0.390 (+0.039)** |
| PART_OF | 0.371 | 0.362 (−0.009) | 0.322 (−0.049) ⚠ |
| TO_DETECT_OR_STUDY | 0.373 | 0.413 (+0.040) | **0.420 (+0.047)** |
| APPLIED_TO | 0.062 | **0.259 (+0.197)** | 0.162 (+0.100) |
| ALTERNATIVE_NAME | 0.131 | 0.173 (+0.042) | 0.127 (−0.004) |

**Class weights ya mejora de verdad** `HAS_CAUSE`/`ASSOCIATED_WITH`/
`TO_DETECT_OR_STUDY` — justo las relaciones que la augmentation no puede
tocar porque no les falta soporte. Y **focal loss dispara `APPLIED_TO`**
(+0.197, el salto más grande de toda la tabla) incluso más que el peso de
clase explícito. `PART_OF` es la excepción a vigilar — con class weights
empeora.

Esto son resultados de una sola seed (42), no multi-seed todavía — se toman
como señal fuerte para decidir el siguiente experimento, no como conclusión
definitiva.

## 8. Qué toca ahora (siguiente paso, todavía no ejecutado)

La vía con más potencial no es generar más datos de las mismas dos clases ni
ampliar a las que no lo necesitan — es **combinar las dos palancas**, que
atacan problemas distintos en relaciones distintas:

1. `eng_train.txt` + `eng_train_llmaug.txt` concatenados (arregla
   `APPLIED_TO`/`ALTERNATIVE_NAME` por falta de datos real)
2. \+ class weights o focal loss en la loss de entrenamiento (arregla
   `HAS_CAUSE`/`ASSOCIATED_WITH`/`TO_DETECT_OR_STUDY`, que no era problema de
   volumen)

Pendiente de decidir: ¿focal loss o class weights para combinar? (dado que
focal loss gana claramente en `APPLIED_TO` pero class weights gana en las
tres relaciones grandes-pero-malas). Y evaluar siempre contra el protocolo
**ciego calibrado**, no solo el dev curado — es la única forma de saber si
esto generaliza de verdad, igual que con cada técnica anterior.



SOBRE LOS DATOS DE ALTERNATIVE_NAME QUE EMPEORA CON DATA 

Miré los 95 pares reales de ALTERNATIVE_NAME uno a uno, y esto explica mucho más de lo que pensaba — no es un patrón homogéneo, es una mezcla de al menos 5 mecanismos lingüísticos distintos:

Fármaco↔abreviatura estándar: topiramate/TPM, valproates/VPA, carbamazepine/CBZ
Fármaco genérico↔nombre comercial: agomelatine/valdoxan, escitalopram/cipralex, hopantenic acid/pantogam
Variantes morfológicas de la misma raíz: epilepsies/epilepsy, myoclonic/myoclonus, diabetic/diabetes, inflammatory/inflamed
Sustantivo anatómico↔forma adjetiva: pulmonary/lung, heart/cardiac, neck/cervical, limb/extremity, cornea/corneal
Sinónimos sueltos de hallazgo clínico: poor prognosis/poor outcome, "significant decrease in pain"/"Reduced pain intensity"
Con esto, "minar pares reales" deja de ser vago — hay dos subtipos que se pueden minar con cero riesgo de invención, porque no dependen de que el LLM afirme un hecho:

Pares anatómicos sustantivo↔adjetivo: es vocabulario médico estándar, cerrado y enumerable (kidney/renal, liver/hepatic, brain/cerebral, stomach/gastric, skin/cutaneous, bone/osseous, nerve/neural, eye/ocular...) — docenas de pares ya conocidos, verificables en cualquier diccionario médico, ninguno inventado.
Variantes morfológicas (sustantivo/adjetivo/verbo de la misma raíz): se pueden derivar con reglas, no hace falta ni LLM para generar el par, solo para escribir la frase.
Para estos dos subtipos, el plan sería: coger pares nuevos (no los 95 que ya hay) de una lista médica estándar, y pedirle al LLM que escriba una frase nueva usando ESE par con marcadores [E1]/[E2] — mismo mecanismo seguro de siempre (filtro de coincidencia exacta), solo que la pareja de entidad ya no se repite.

Los otros 3 subtipos (fármaco↔marca, sinónimos sueltos) son más arriesgados de minar sin una fuente externa fiable — los dejaría fuera por ahora.

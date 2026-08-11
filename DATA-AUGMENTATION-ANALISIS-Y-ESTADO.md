# Data Augmentation con LLM — análisis de impacto esperado y estado de ejecución

Fecha: 2026-08-05

Este documento complementa a `PLAN-DATA-AUGMENTATION-LLM.md` (que explica el
qué y el porqué del diseño). Aquí se responde a dos preguntas concretas que
surgieron después de tener el plan montado — **cuánto podría mejorar el macro
F1 de verdad** y **cuántas instancias hacen falta**, con evidencia de los
experimentos ya corridos, no a ojo — y se deja constancia de qué pasó al
intentar ejecutar la generación por primera vez de verdad.

## 1. Qué LLM usar

**Gemini `gemini-3.5-flash`, capa gratuita.** Confirmado con una prueba en
vivo: 11/11 instancias de prueba (`APPLIED_TO` y `ALTERNATIVE_NAME`) pasaron
el pipeline completo (llamada + parseo de marcadores `[E1]/[E2]` + filtro de
coincidencia exacta de entidades) sin fallos de calidad. No hay motivo para
cambiar a Claude o GPT — costarían dinero por token sin necesidad; el
problema real no es de calidad del LLM (ver sección 3).

## 2. Cuánto podría subir el macro F1 — techo realista, no una promesa

Macro F1 es la media simple de 14 relaciones (sin contar `no_relation`), así
que cada una pesa 1/14 ≈ 0.071 del total. Los dos peores casos ahora mismo
(PubMedBERT, multi-seed):

| Relación | F1 actual | Precision | Recall | Soporte (dev) |
|---|---|---|---|---|
| ALTERNATIVE_NAME | 0.346 | 0.759 | **0.224** | 98 |
| APPLIED_TO | 0.460 | 0.606 | **0.370** | 54 |

El patrón (precision razonable, recall muy bajo) es la firma de **infra-
predicción**, no de confusión sistemática irreparable: el modelo casi no se
atreve a predecir estas clases. Confirmado mirando la matriz de confusión —
`ALTERNATIVE_NAME` se predice como `no_relation` en 50 de 98 casos (más de la
mitad); `APPLIED_TO` se confunde sobre todo con `TO_DETECT_OR_STUDY` (19) y
`no_relation` (13). Es exactamente el síntoma que el data augmentation
debería aliviar: dar más ejemplos positivos para que el modelo se atreva a
predecir la clase con más confianza.

**Estimación de mejora si funciona bien:**
- ALTERNATIVE_NAME: 0.346 → ~0.65 (nivel medio del resto) → +0.304 × (1/14) ≈ **+0.022** al macro
- APPLIED_TO: 0.460 → ~0.65 → +0.19 × (1/14) ≈ **+0.014** al macro
- Combinado: **+0.03 a +0.04 de macro F1**, si el augmentation funciona como se espera en estas dos clases.

Para poner ese número en contexto: el ruido entre seeds que medimos en el
experimento multi-seed (1G) es de **±0.01 a ±0.02** en ciego calibrado. Un
+0.03-0.04 sería una ganancia **real** por nuestra propia regla (2-3× la
std), no un espejismo de una sola corrida — pero tampoco hay que esperar una
transformación dramática del modelo. Es una mejora modesta pero medible, no
un salto de nivel.

## 3. Cuántas instancias hacen falta — evidencia, no intuición

Mirando qué soporte necesitan las clases que YA funcionan bien en el mismo
experimento: `ABBREVIATION` tiene solo 74 ejemplos en dev y saca F1=0.920.
Eso indica que del orden de **75-100 ejemplos bastan** para que una clase se
aprenda bien, **si la clase es intrínsecamente aprendible** (patrones de
superficie razonablemente claros — que parece ser el caso tanto de
`ALTERNATIVE_NAME` como de `APPLIED_TO`, dado que el error dominante es
infra-predicción y no confusión masiva con una clase concreta).

El plan ya lleva `ALTERNATIVE_NAME` de 95→285 y `APPLIED_TO` de 43→172 —
ambas quedan muy por encima de ese umbral de 75-100, con margen de sobra. Es
un objetivo razonable respaldado por lo que ya observamos en el propio
dataset, no una cifra sacada de la nada.

## 4. El riesgo de que no generalice — real, y cómo se vigila

`ALTERNATIVE_NAME` solo tiene 95 frases **originales**. Generar 190 variantes
de esas 95 no da 190 hechos independientes — le da al modelo 95 conceptos con
2-3 disfraces cada uno (mismas entidades, estructura de frase parecida,
solo cambia el "relleno" parafraseado). El riesgo concreto es que el modelo
aprenda "el estilo de parafraseo de Gemini" en vez del patrón semántico real
de la relación, y que la mejora sea un espejismo de sobreajuste al dev
curado sin trasladarse al protocolo ciego real.

Por eso el paso de la sección 7 del plan original (entrenar con datos
aumentados + comparar contra el baseline en el protocolo **ciego
calibrado** oficial, no solo en dev curado) no es opcional — es la única
forma de medir si esto generaliza de verdad. Ningún número de esta
generación se da por bueno hasta que no pase por esa comparación, igual que
ya pasó con `neg_ratio`, `focal loss` y `class weights` (sección 1 del plan
original): sonar razonable no es lo mismo que funcionar en la práctica.

## 5. Estado real de la ejecución — bloqueo encontrado

Al lanzar la generación completa (9 relaciones, ~1350 instancias objetivo) se
descubrió algo que **el plan original no tenía en cuenta**:

**La capa gratuita de `gemini-3.5-flash` no tiene un límite por minuto — tiene
un límite de 20 peticiones al DÍA** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
`quotaValue: 20`). Esto explica dos cosas:

- Por qué el primer intento de generación (antes de este documento) solo
  produjo 19 instancias en vez de las ~1350 esperadas — justo por debajo del
  límite diario.
- Por qué el segundo intento (relanzado en `tmux` para sobrevivir a cortes de
  sesión) se quedó atascado reintentando 429 (`RESOURCE_EXHAUSTED`) una y
  otra vez: 420/980 instancias procesadas, pero solo 11 generadas de verdad
  — el resto son reintentos contra una cuota que no se reabre hasta el día
  siguiente.

**Bug adicional encontrado y corregido:** el script guardaba todas las
instancias generadas en memoria y solo las escribía a disco al final
(`with open(out_path, "w") as f: ...`, sin `flush`). Al matar el proceso
atascado en la cuota, **se perdieron las 11 instancias que sí se habían
generado** correctamente esa sesión — nunca llegaron a tocar disco. Corregido
en `baseline/augment_llm.py`:
- Cada escritura ahora hace `f.flush()` inmediatamente (nada se queda solo en
  memoria).
- El fichero se abre en modo `"a"` (append) en vez de `"w"`, para que
  relanzar el script en días sucesivos (una vez se reponga la cuota) vaya
  acumulando en vez de borrar lo generado en sesiones anteriores.

## 6. Qué hacer a partir de aquí

Con 20 peticiones/día de cuota gratuita y ~1350 llamadas necesarias, generar
todo el plan tal cual llevaría **~68 días** a este ritmo — inviable. Opciones
reales, sin decidir todavía cuál tomar:

1. **Esperar y repartir en varios días** (20/día): con la clase prioritaria
   primero (`ALTERNATIVE_NAME`, ~190 llamadas ≈ 10 días solo para esa),
   dejando `APPLIED_TO` para después. Gratis, pero lento — no cuadra con
   "generar ya" que era el objetivo de hoy.
2. **Activar facturación en el proyecto de Gemini** (sigue siendo muy barato
   — `gemini-3.5-flash` cuesta fracciones de céntimo por instancia a precio
   de pago): las ~1350 llamadas costarían del orden de unos pocos dólares,
   no hay que cambiar nada del código, solo el plan de facturación en Google
   AI Studio.
3. **Repartir entre varias API keys / proyectos gratuitos** (cada proyecto de
   Google Cloud tiene su propia cuota de 20/día) — funciona pero es más
   trabajo de gestión y roza el espíritu de los límites del servicio.
4. **Reducir el objetivo** a solo las dos clases que de verdad mueven la
   aguja del macro F1 (sección 2): `ALTERNATIVE_NAME` (~190 llamadas) y
   `APPLIED_TO` (~129 llamadas) — unas 320 llamadas en total, ~16 días
   gratis o unos céntimos de pago. Coherente con el hallazgo de que estas
   dos son las que de verdad justifican el esfuerzo.

La opción 4 combinada con activar facturación (opción 2) parece la más
sensata: es barato, y ya sabemos por la sección 2 que ampliar solo estas dos
clases es lo que realmente puede mover el macro F1 de forma medible.

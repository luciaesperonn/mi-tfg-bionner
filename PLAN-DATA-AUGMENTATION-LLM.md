# Plan: Data Augmentation con LLM (Gemini) para las clases minoritarias

Fecha: 2026-07-30

Este documento es básicamente para dejar constancia de qué vamos a hacer, por
qué lo hacemos y cómo lo hemos montado, antes de lanzar nada. Así si dentro de
unas semanas releo esto (o alguien más lo lee) se entiende el razonamiento
sin tener que reconstruirlo de memoria.

## 1. El problema que estamos intentando arreglar

Ya sabemos (por los experimentos 1G, 4A, 4B, 4C y por `HALLAZGOS-BUGS-TOKENIZACION.md`)
que el dataset tiene un desbalance de clases bastante bruto. Si contamos
cuántos ejemplos hay de cada relación en el train actual (después de aplicar
`neg_ratio=3`, o sea contando también los `no_relation`):

| relación | nº ejemplos |
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
| APPLIED_TO | 43 |

Fijaos en la diferencia entre `SUBCLASS_OF` (743) y `APPLIED_TO` (43): 17
veces más ejemplos. Y la métrica que usamos para evaluar el modelo (macro F1)
le da el MISMO peso a cada clase, le dé igual si tiene 743 o 43 ejemplos. Eso
significa que las clases pequeñas son las que más están penalizando la nota
final, porque el modelo apenas ha visto ejemplos suyos para aprender a
reconocerlas bien.

Ya intentamos arreglar esto de otras formas antes:

- **`neg_ratio`** (cuántos negativos por cada positivo en el train): esto
  ayuda a la proporción positivo/negativo en general, pero no toca el
  desbalance ENTRE las clases positivas -- `APPLIED_TO` sigue teniendo 43
  ejemplos le pongas el `neg_ratio` que le pongas.
- **Class weights** (experimento 4C, todavía corriendo mientras escribo
  esto): le dice a la loss "un fallo en `APPLIED_TO` cuenta más que un fallo
  en `SUBCLASS_OF`", pero sigue entrenando con los mismos 43 ejemplos de
  siempre. No crea información nueva, solo redistribuye la importancia de la
  que ya hay.
- **Focal loss** (experimento 4B): parecido, reescala el gradiente según lo
  fácil o difícil que sea cada ejemplo, pero tampoco genera ejemplos nuevos.

Ninguna de las tres soluciona el problema de raíz, que es que hay clases con
muy pocos ejemplos de los que aprender. Para eso hace falta **crear ejemplos
nuevos** -- eso es exactamente lo que es "data augmentation".

## 2. Qué opciones había para crear ejemplos nuevos

Antes de decidirnos, hablamos de varias formas de hacer data augmentation
para relation extraction:

1. **Entity replacement**: coger una frase ya existente y cambiar la entidad
   head o tail por otra entidad del mismo tipo semántico, sacada de otra
   instancia con la misma relación. Es barato y no necesita ningún LLM, pero
   el riesgo es que la entidad nueva no encaje bien en el contexto de la
   frase original y la relación deje de ser verdadera aunque los tipos
   coincidan (ej. meter un fármaco en una frase que originalmente hablaba de
   un gen).
2. **Back-translation** (traducir a otro idioma y volver a traducir):
   genera más variedad de superficie textual, pero recalcula los spans de
   entidad desde cero -- y ya sabemos que ahí es justo donde hemos tenido
   bugs de verdad (ver `HALLAZGOS-BUGS-TOKENIZACION.md`), así que añadir más
   pasos de recálculo de spans automáticos nos parecía arriesgado.
3. **Parafraseo con LLM manteniendo las entidades fijas**: le pides al LLM
   que reescriba la frase con otras palabras, pero le dices explícitamente
   que las dos menciones de entidad tienen que quedar EXACTAMENTE igual, letra
   por letra. Así no hay que recalcular nada raro de traducción y el LLM solo
   tiene que cambiar el "relleno" de la frase, no las entidades.

Al final decidimos ir con la opción 3 (parafraseo con LLM), porque:

- Genera ejemplos realmente nuevos (frases distintas), no solo recombina lo
  que ya había como el entity replacement.
- Al obligar a que las entidades queden literales, evitamos el problema de
  "quién sabe si el LLM tradujo bien el nombre de la entidad" que tendría el
  back-translation.
- El propio LLM puede marcar dónde empieza y termina cada entidad en su
  respuesta (ver sección 4), así que no dependemos de que el LLM sepa contar
  caracteres para darnos los offsets -- los calculamos nosotros con código
  normal, de forma determinista.

## 3. Qué LLM usamos y por qué

Al principio iba a usar la API de Claude (Anthropic), pero cuesta dinero por
token. Como esto es para un TFG y no queremos gastar, cambiamos a **Gemini**
(Google), que tiene una capa gratuita real (sin tarjeta de crédito) para
algunos modelos. Usamos concretamente `gemini-3.5-flash`, que a día de hoy
(mediados de 2026) está en esa capa gratuita.

Cosas a tener en cuenta con la capa gratuita:

- El límite de peticiones por minuto (RPM) es bajo, así que el script mete
  una pausa de 2 segundos entre llamadas para no pasarnos, y reintenta con
  espera creciente si Gemini nos devuelve un error de límite.
- Al ser gratis, generar 1000-1500 frases nuevas no cuesta nada de dinero,
  pero sí tarda un rato en tiempo real (calculamos más o menos 1 hora, ver
  sección 5).

## 4. Cómo funciona el script técnicamente (`baseline/augment_llm.py`)

Para cada instancia positiva que queremos aumentar (una frase + su entidad
head + su entidad tail + la relación entre ellas):

1. Le pedimos a Gemini que reescriba la frase con otras palabras, pero
   manteniendo:
   - La misma relación semántica entre las dos entidades (que no se quede
     una frase ambigua o que ya no la sostenga).
   - Las dos menciones de entidad EXACTAMENTE igual (ni traducir, ni
     abreviar distinto, ni cambiar el número).
2. Le pedimos que en su respuesta marque la entidad head con
   `[E1] ... [/E1]` y la entidad tail con `[E2] ... [/E2]`, alrededor del
   texto literal de cada una.
3. Con esa respuesta, nuestro código (no el LLM) quita los marcadores y
   calcula en qué posición de caracter empieza y termina cada entidad en el
   texto final -- así el offset es 100% correcto porque lo hemos calculado
   nosotros contando caracteres de verdad, no confiando en que el LLM lo
   haga bien.
4. **Filtro de calidad**: si el texto que queda dentro de `[E1]...[/E1]` no
   coincide exactamente (ignorando mayúsculas) con la entidad head original,
   descartamos esa generación entera. Lo mismo para la tail. Esto es
   importante porque así nos aseguramos de que, aunque el LLM se invente
   alguna tontería o cambie algo que no debía, esa instancia mala no entra en
   el dataset -- se tira directamente, no se "arregla a medias".
5. Si pasa el filtro, guardamos la nueva instancia en el mismo formato JSON
   que usa el resto del proyecto (mismo esquema que `eng_train.txt`): texto,
   posición de head, posición de tail, relación, y un `doc_id` con el sufijo
   `_llmaug` para que se note que es una instancia sintética y no un
   documento real de BioNNE-R.

Lo que NO hace este script: no valida que el LLM haya entendido bien la
relación (nos fiamos del prompt + de que el filtro de entidades pase). Es
decir, comprobamos que las entidades sean correctas, pero no hay forma
automática de comprobar que la relación semántica siga siendo 100% verdad --
eso es un riesgo que asumimos conscientemente (ver sección 6).

## 5. Qué relaciones vamos a aumentar y cuánto

Como decíamos en la sección 1, el problema es que hay clases con muy pocos
ejemplos comparadas con las grandes (`SUBCLASS_OF`, `HAS_CAUSE`, `AFFECTS`,
`ASSOCIATED_WITH`). Esas cuatro las dejamos tal cual están, no las tocamos.

Para el resto, la idea es generar suficientes variantes por instancia para
que cada clase se acerque a un "nivel objetivo" de más o menos 150-300
ejemplos (en vez de generar el mismo número de variantes para todas, que no
tendría mucho sentido si una clase ya tiene 200 y otra solo 43):

| relación | ejemplos originales | variantes por instancia | ejemplos nuevos (estimado) | total aprox. tras aumentar |
|---|---|---|---|---|
| APPLIED_TO | 43 | 3 | ~129 | ~172 |
| USED_IN | 89 | 2 | ~178 | ~267 |
| ALTERNATIVE_NAME | 95 | 2 | ~190 | ~285 |
| ABBREVIATION | 97 | 2 | ~194 | ~291 |
| TREATED_USING | 101 | 1 | ~101 | ~202 |
| ORIGINS_FROM | 120 | 1 | ~120 | ~240 |
| FINDING_OF | 121 | 1 | ~121 | ~242 |
| PHYSIOLOGY_OF | 150 | 1 | ~150 | ~300 |
| TO_DETECT_OR_STUDY | 164 | 1 | ~164 | ~328 |
| PART_OF | 204 | 0 (no se toca) | 0 | 204 |

Total estimado de instancias nuevas generadas: **~1350**, aunque el número
real será algo menor porque el filtro de calidad de la sección 4 va a
descartar algunas generaciones (esperamos que la mayoría pasen, pero no
todas).

Con la pausa de 2 segundos entre llamadas por el límite de la capa gratuita,
esto son del orden de 1350 llamadas × ~2.5s ≈ **entre 45 minutos y 1 hora**
de ejecución. Por eso lo vamos a lanzar en segundo plano.

El resultado se guarda en un fichero nuevo, **separado** del train original:
`data/english/eng_train_llmaug.txt`. No tocamos `eng_train.txt` para nada --
así el dataset "oficial" con el que hemos validado todos los experimentos
anteriores (1A, 3A, 4A, 4B, 4C...) queda intacto, y el fichero aumentado es
un añadido opcional que se puede concatenar con el original cuando queramos
probar un experimento nuevo con él.

## 6. Riesgos y cosas que hay que tener en cuenta (honestamente)

- **Ruido de etiqueta**: aunque filtramos que las entidades coincidan, no
  podemos garantizar al 100% que la relación semántica siga siendo
  exactamente la misma en la frase parafraseada. Es un riesgo real, no algo
  que hayamos resuelto del todo.
- **Esto no sustituye a arreglar los bugs de tokenización**: los datos
  aumentados se generan a partir del train original (`eng_train.txt`), que
  todavía tiene el problema de entidades anidadas documentado en
  `HALLAZGOS-BUGS-TOKENIZACION.md`. El fix de `patch_opennre.py` se aplica
  en el momento de entrenar (vía `fix_entity_markers()`), así que esto sigue
  funcionando igual de bien con los datos aumentados que con los originales.
- **No sabemos todavía si esto mejora el modelo de verdad**: generar los
  datos es solo el primer paso. Hace falta un experimento real (entrenar con
  train original + train aumentado y comparar contra el baseline 3A en el
  protocolo blind oficial) para saber si el macro F1 sube, se queda igual o
  incluso baja. Como con neg_ratio, focal loss y class weights: no se da por
  hecho que algo "suene razonable" vaya a funcionar en la práctica hasta que
  no se mide.
- **Una sola pasada del LLM, no hay revisión humana**: no vamos a leer las
  ~1350 frases generadas una por una para comprobar que tienen sentido. Nos
  fiamos del filtro automático (coincidencia exacta de las entidades) como
  única red de seguridad.

## 7. Qué toca después de generar los datos

Este documento cubre solo la generación de los datos aumentados. Una vez
generados, el siguiente paso lógico (no incluido en este documento) sería
montar un experimento nuevo tipo `3A`/`4A`/`4B`/`4C` que entrene con
`eng_train.txt` + `eng_train_llmaug.txt` concatenados y lo compare contra el
baseline `3A` en el protocolo blind oficial (argmax + calibración fina), para
ver si de verdad ayuda o no.

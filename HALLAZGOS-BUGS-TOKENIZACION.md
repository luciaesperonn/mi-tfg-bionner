# Hallazgos: dos bugs de tokenización en el pipeline OpenNRE

Documento de constancia para la memoria del TFG. Describe dos problemas
encontrados el 2026-07-29 en el mecanismo de marcado de entidades de
`opennre.encoder.BERTEntityEncoder` (usado por **todos** los experimentos del
proyecto: 1A, 1B, 1C, 1D, 1G y los posteriores), cómo se detectaron, su
impacto medido, y el arreglo aplicado.

Ninguno de los dos bugs tiene relación con "typed entity markers" (la técnica
que se estaba probando cuando se encontraron) — son problemas de la capa base
de tokenización, presentes desde el primer experimento del proyecto.

---

## Contexto: cómo se detectaron

Al entrenar PubMedBERT con typed entity markers (notebook `2A`, ver más abajo)
el resultado en blind calibrado salió **peor** que el baseline sin typed
markers (0.4134 vs 0.4316), cuando se esperaba una mejora. Para entender por
qué, se inspeccionaron directamente los tokens que recibía el modelo (en vez
de fiarse solo del número de macro F1 final) — ahí aparecieron ambos
problemas.

---

## Bug 1 — `tokenize()` duplica texto cuando las entidades están anidadas

### Qué pasaba antes

`BERTEntityEncoder.tokenize()` (en el paquete `opennre`, no en este repo)
asume que la entidad head y la entidad tail **nunca se solapan**: parte la
frase en 5 trozos asumiendo que van una detrás de otra en el texto —
`sent0 + ent0 + sent1 + ent1 + sent2`, donde `ent0` es la entidad que empieza
antes y `ent1` la que empieza después.

Cuando una entidad está **anidada dentro de la otra** (p. ej. "lung" dentro de
"lung injury" — muy habitual en relaciones tipo AFFECTS/PART_OF/SUBCLASS_OF,
que conectan un término específico con el término general que lo contiene),
esa asunción se rompe en silencio: el trozo `sent1` se vuelve un slice vacío
o negativo, y el trozo de texto que queda "dentro" de la entidad exterior se
duplica en la secuencia final.

**Ejemplo real** (`data/english/eng_dev.txt`, `head="lung injury"` en
`[114,125]`, `tail="lung"` en `[114,118]`):

```
texto real:
  "...diagnose lung injury in pregnant women with blood cancers."

tokens que recibía el modelo (ANTES del fix):
  ... diagnose [X] lung injury [X] [X] lung [X] injury in pregnant women ...
                                              ^^^^^^ "injury" duplicado
```

### Impacto medido

| conjunto | % de instancias con entidades anidadas/solapadas |
|---|---|
| `eng_train.txt` | 13.4% (1 703 / 12 739) |
| `eng_dev.txt` | 51.0% (1 514 / 2 967) |
| pool completo de candidatos blind (282 364) | 1.6% |
| **relaciones GOLD reales dentro del blind (2 891)** | **51.0% (1 474 / 2 891)** |

El pool de candidatos blind está dominado por negativos (`no_relation`), por
eso el porcentaje global es bajo — pero de las relaciones que **sí** cuentan
para el macro F1, la mitad le llegaban al modelo con la frase corrompida.
Esto afecta a entrenamiento (13.4% de train) y a evaluación (mitad del gold
real) de **todos** los experimentos hechos hasta ahora con este pipeline.

### Qué se hizo

Se reescribió `tokenize()` para no asumir 3 regiones fijas: en vez de eso,
corta la frase por la unión de los 4 bordes de entidad (`h_start, h_end,
t_start, t_end`), tokeniza cada segmento resultante una vez, e inserta los
marcadores en los puntos de corte donde abre/cierra cada entidad (los cierres
van antes que las aperturas en caso de empate, para no anidar un marcador al
revés). Para el caso sin solape esto da exactamente la misma segmentación que
el código original (verificado token a token, byte a byte, sobre varios
ejemplos reales); para el caso anidado, produce la misma secuencia que
tokenizar la frase entera de una vez, sin duplicar ni perder palabras.

- Implementación: `baseline/patch_opennre.py::fix_nested_entity_tokenize()`
  (monkeypatch, mismo estilo que `add_macro_f1_metric()` ya existente).
- Verificado con: casos sintéticos (anidamiento en ambos sentidos, entidades
  adyacentes, orden invertido) + comparación directa contra el `tokenize()`
  original en casos no anidados + el ejemplo real de arriba.
- **Estado: arreglado y verificado, pero a fecha de este documento todavía no
  se ha usado en ningún entrenamiento completo** (ver notebook `3A` para el
  primer uso real).

---

## Bug 2 — los marcadores `[unused0]-[unused3]` son invisibles en 3 de 4 encoders

### Qué pasaba antes

Para decirle al modelo dónde empieza y termina cada entidad, OpenNRE inserta
4 tokens especiales alrededor de head y tail: `[unused0]`/`[unused1]` para la
cabeza, `[unused2]`/`[unused3]` para la cola. BERT reserva los ids 1-99 como
huecos "`[unused]`" sin uso, pensados exactamente para casos como este.

El problema: los tokenizers de dominio biomédico (PubMedBERT, BioLinkBERT,
BioBERT) **reconstruyen el vocabulario desde cero** a partir de literatura
médica, y en esa reconstrucción esos huecos dejan de ser huecos reales —
varios de ellos (o todos) terminan apuntando al mismo id que `[UNK]`
("palabra desconocida").

Comprobado directamente pidiéndole a cada tokenizer el id de cada marcador:

| Encoder | `[unused0]` | `[unused1]` | `[unused2]` | `[unused3]` | `[UNK]` | Estado |
|---|---|---|---|---|---|---|
| PubMedBERT (BiomedNLP-BiomedBERT-base) | 1 | 1 | 1 | 1 | 1 | **roto — los 4 = `[UNK]`** |
| BioLinkBERT-base | 1 | 1 | 1 | 1 | 1 | **roto — los 4 = `[UNK]`** |
| BioBERT (dmis-lab/biobert-v1.1) | 100 | 1 | 2 | 3 | 100 | **roto — el de "abre cabeza" = `[UNK]`** |
| SciBERT (allenai/scibert_scivocab_uncased) | 1 | 2 | 3 | 4 | 101 | funciona: los 4 distintos entre sí y de `[UNK]` |

### Impacto

Con los 4 marcadores colapsados al mismo id, el modelo no puede distinguir
"aquí empieza la cabeza" de "aquí termina la cola" ni de una palabra
cualquiera fuera de vocabulario en medio de la frase — son, a nivel de
embedding de entrada, el mismo token. Afecta a **todos** los entrenamientos
hechos con PubMedBERT, BioLinkBERT-base o BioBERT en este proyecto (3 de los
4 encoders comparados en el estudio multiseed). SciBERT es el único encoder
del proyecto donde este mecanismo funcionaba tal como está diseñado.

Esto es un confusor de fondo para toda comparación entre encoders hecha hasta
ahora: la ventaja relativa de SciBERT en varias métricas podría deberse en
parte a ser el único con el marcado de entidades intacto, no (solo) a la
calidad de su preentrenamiento.

### Qué se hizo

En vez de depender de que `[unused0-3]` sean huecos reales en cada
vocabulario (asunción que no se cumple), se añaden 4 tokens especiales
**nuevos y dedicados** (`[E1]`, `[/E1]`, `[E2]`, `[/E2]`) a cada tokenizer con
`tokenizer.add_special_tokens(...)`, y se redimensiona la matriz de
embeddings del modelo a juego con `bert.resize_token_embeddings(...)` — la
forma estándar de añadir vocabulario nuevo a un modelo preentrenado,
funciona sin importar qué huecos tuviera el vocabulario original. Los
embeddings nuevos se inicializan a partir de la media/covarianza de los
existentes y se entrenan desde ahí durante el fine-tuning normal.

- Implementación: `baseline/patch_opennre.py::fix_entity_markers()` —
  incluye también el fix del Bug 1 (mismo `tokenize()` nuevo). Sustituye a
  `fix_nested_entity_tokenize()` cuando se quieren los dos arreglos a la vez.
- Verificado con: los 4 ids nuevos son distintos entre sí y de `[UNK]`
  (30522-30525), la matriz de embeddings queda redimensionada a juego
  (30528 filas), y un forward pass real del modelo con el vocabulario
  ampliado no falla.
- **Estado: arreglado, verificado a nivel de código, y ya probado en un
  entrenamiento real** (notebook `3A`, PubMedBERT seed 42, sin typed markers,
  mismos hiperparámetros que el baseline). Resultado:

  | | argmax | calibrado (fino) |
  |---|---|---|
  | baseline (con los 2 bugs) | 0.3162 | 0.4316 |
  | + `fix_entity_markers()` | 0.3338 | 0.4298 |
  | delta | **+0.0175** | **-0.0018** |

  Mejora clara en argmax sin calibrar; en calibrado el cambio es minúsculo
  (dentro del ruido de seed a seed medido en el estudio multiseed, std≈0.006-
  0.012) — con una sola seed no se puede afirmar que arreglar los bugs
  "no cambia nada" ni que "mejora": hace falta repetir con varias seeds para
  saber si el efecto real es distinto de cero. Lo que sí queda demostrado es
  que el fix funciona correctamente a nivel de tokens (verificado dentro del
  propio notebook antes de entrenar) y no rompe el entrenamiento.

---

## Qué queda pendiente

- El notebook `3A` reentrena PubMedBERT (mismos hiperparámetros que el
  baseline 1A/1G, seed 42) con **ambos fixes aplicados**, sin typed markers,
  para aislar el efecto de arreglar los bugs en sí — el número resultante
  debería ser la referencia correcta de PubMedBERT a partir de ahora.
- Para que la comparación entre los 4 encoders sea válida de nuevo, habría
  que repetir el estudio multiseed (1G) con `fix_entity_markers()` aplicado
  en los 4 — no se ha hecho todavía por el coste (~5h en esta GPU).
- El experimento de typed markers (notebooks `2A`/`2B`) se hizo ANTES de
  encontrar y arreglar estos bugs — sus resultados numéricos siguen siendo
  válidos como comparación relativa dentro del mismo notebook (baseline y
  técnica sufren el mismo problema de fondo por igual), pero no reflejan el
  techo real de cada modelo. Repetirlo sobre una base ya arreglada es un
  trabajo futuro razonable.

Modelo	curado	ciego argmax	ciego calibrado	threshold
DeBERTa-v3-base	0.7649	0.2885	0.3741	0.96
XLNet-base	0.7292	0.2732	0.3430	0.99 ⚠borde
SciFive-T5-base	0.7169	0.1729	0.2251	0.25
Los tres quedan claramente por debajo de los 4 que ya tenemos — hasta BioBERT (el peor de los 4, 0.408 en ciego calibrado con multi-seed) les gana a los tres. SciFive-T5 en particular se hunde mucho (0.225): tiene sentido, es un modelo generativo (seq2seq), no un clasificador — encaja peor con este tipo de tarea, y su threshold óptimo (0.25) es rarísimo comparado con el resto (todos entre 0.90-0.99), señal de que se comporta de forma distinta, probablemente peor calibrado.

Dado lo grande de la diferencia frente a los 4 ya estudiados (BioBERT, el peor de esos, saca 0.408 — muy por encima de estos tres), por la misma regla que aplicamos antes ("si la diferencia es clara, no hace falta multi-seed para confirmarlo") yo no invertiría en repetir estos tres con 3 seeds — parece que se quedan fuera de la comparación central sin necesidad de más pruebas. La excepción es XLNet, cuyo threshold está otra vez pegado al borde 0.99 — su número real podría subir un poco, pero probablemente no lo suficiente para acercarse al grupo de los 4.

¿Quieres que corra el barrido fino también sobre estos 3 para confirmar sus verdaderos óptimos, o los dejamos así dado lo claro de la diferencia?
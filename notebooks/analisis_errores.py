"""
Análisis de errores del baseline BioNNE-R
Compara predicciones vs gold y muestra dónde falla el modelo
"""
import pandas as pd
import collections

# Cargar predicciones y gold
pred = pd.read_csv("../outputs/eng_pred.tsv", sep="\t")
gold = pd.read_csv("../data/en/dev/eng-dev-rel.tsv", sep="\t")

print(f"Predicciones: {len(pred)}")
print(f"Gold: {len(gold)}")

# Crear clave única por instancia
def clave(row):
    return f"{row['document_id']}|{row['head_span']}|{row['tail_span']}"

gold["clave"] = gold.apply(clave, axis=1)
pred["clave"] = pred.apply(clave, axis=1)

gold_dict = dict(zip(gold["clave"], gold["relation"]))
pred_dict = dict(zip(pred["clave"], pred["relation"]))

# Encontrar errores
errores = []
for clave_inst, gold_rel in gold_dict.items():
    pred_rel = pred_dict.get(clave_inst, "NO_PREDICHA")
    if gold_rel != pred_rel:
        errores.append({
            "gold": gold_rel,
            "predicha": pred_rel,
        })

print(f"\nTotal errores: {len(errores)} de {len(gold_dict)}")
print(f"Tasa de error: {100*len(errores)/len(gold_dict):.1f}%")

# Confusiones más frecuentes
print("\nCONFUSIONES MÁS FRECUENTES:")
print(f"{'Gold':25s} {'Predicha':25s} {'Veces':>6}")
print("-" * 60)
confusiones = collections.Counter(
    (e["gold"], e["predicha"]) for e in errores
)
for (gold_rel, pred_rel), n in confusiones.most_common(15):
    print(f"{gold_rel:25s} {pred_rel:25s} {n:>6}")
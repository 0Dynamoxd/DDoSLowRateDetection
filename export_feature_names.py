import os
import json
import pandas as pd

TARGET = "label"

BASE = os.path.join("cicddos", "processed")
TRAIN_CSV = os.path.join(BASE, "train_lowrate.csv")
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"No se encontró {TRAIN_CSV}")

# Leer solo una fila (no carga todo el archivo)
df = pd.read_csv(TRAIN_CSV, nrows=1)
features = [c for c in df.columns if c != TARGET]

out_path = os.path.join(ART_DIR, "feature_names_lowrate.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(features, f, indent=2)

print(f"Guardadas {len(features)} features en {out_path}")

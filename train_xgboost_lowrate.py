import os, json, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split


BASE = "cicddos/processed"
TRAIN_CSV = os.path.join(BASE, "train_lowrate.csv")
TEST_CSV  = os.path.join(BASE, "test_lowrate.csv")
ART_DIR   = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

TARGET = "label"


train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

features = [c for c in train.columns if c != TARGET]

X_train = train[features]
y_train = train[TARGET]
X_test  = test[features]
y_test  = test[TARGET]

print(f"Loaded dataset: {len(X_train)} train | {len(X_test)} test | {len(features)} features")


# DMatrix

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=features)


# Split for early stopping (XGBoost doc recomienda 80/20 interno)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            stratify=y_train, random_state=42)
dtr  = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)



neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = float(neg) / float(max(pos, 1))

params = {
    "objective": "binary:logistic", #clasificación binaria
    "eval_metric": "aucpr",      # AUC-PR  área bajo la curva Precision-Recall
    "learning_rate": 0.05,       # eta Dicta qué tan rápido aprende el modelo.
    "max_depth": 6, #profundidad máxima de cada árbol 6 equilibrado , alto memoriza datos overfitting si es bajo underfitting 
    "min_child_weight": 3, #número mínimo de muestras necesarias para crear un nuevo nodo en el árbol
    "subsample": 0.8, #El modelo usará solo el 80% de los datos (filas) de forma aleatoria para construir cada árbol.
    "colsample_bytree": 0.8, #El modelo usará solo el 80% de las variables (columnas) para cada árbol
    "reg_lambda": 1.0, #Penaliza los pesos de las hojas para que no sean demasiado grandes
    "tree_method": "hist",       # "gpu_hist" PARA TARJETA GRAFICA
    "scale_pos_weight": scale_pos_weight, #Le da más "importancia" o peso a la clase minoritaria.
    "random_state": 42 #mismo resultado para cada vez que se manda el codigo
}


# Entrenamiento

print(" Entrenando modelo XGBoost (80/20 con early stopping)...")
evals = [(dtr, "train"), (dval, "eval")]

bst = xgb.train(
    params=params,
    dtrain=dtr,
    num_boost_round=4000,
    evals=evals,
    early_stopping_rounds=80,
    verbose_eval=100
)

print(f" Best iteration: {bst.best_iteration + 1}")


# Evaluación en test
y_prob = bst.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
roc  = roc_auc_score(y_test, y_prob)
pr   = average_precision_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n Métricas en Test:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc:.4f}")
print(f"PR-AUC    : {pr:.4f}")
print("Confusion Matrix [0,1]:\n", cm)
print("\nReporte detallado:\n", classification_report(y_test, y_pred, digits=4))


# Importancias de características
importance = bst.get_score(importance_type="gain")
imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
top10 = imp_sorted[:10]
print("\n🔝 Top-10 features más importantes:")
for f, v in top10:
    print(f"{f:25s} {v:.5f}")


# Guardar artefactos
bst.save_model(os.path.join(ART_DIR, "xgb_lowrate_model.json"))

with open(os.path.join(ART_DIR, "xgb_lowrate_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc, "pr_auc": pr,
        "confusion_matrix": cm.tolist(),
        "best_iteration": int(bst.best_iteration)
    }, f, indent=2)

with open(os.path.join(ART_DIR, "xgb_lowrate_importance.json"), "w", encoding="utf-8") as f:
    json.dump(imp_sorted, f, indent=2)

print(f"\n Modelo y métricas guardados en: {ART_DIR}")

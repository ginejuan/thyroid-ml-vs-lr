"""
03_random_forest.py
-------------------
Random Forest con tuneo de mtry (max_features) y min_samples_leaf por
una CV interna pequeña (5 folds) en cada fold externo. Para mantener el
coste razonable, ajustamos primero el grid en el dataset completo y luego
usamos esa configuración para la CV externa repetida (10×5).

También calibramos las probabilidades isotónicamente (las RF tienden a
sub-confianza en los extremos), reportando el modelo CALIBRADO.
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils_cv import (load_data, cv_predict_proba, evaluate,
                      save_metrics_and_preds, SEED, N_FOLDS, N_REPEATS)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados"
X, y, _ = load_data(ROOT)
print(f"n = {len(y)}, casos = {y.sum()} ({y.mean()*100:.1f}%)")

# 1. Tuneo de hiperparámetros (CV 5-fold interna) -------------------------
param_grid = {
    "n_estimators":     [500],
    "max_features":     [3, 4, 5, 6, "sqrt"],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_depth":        [None],
}
print("\nBúsqueda de hiperparámetros (GridSearchCV 5 folds, scoring=AUC)...")
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight=None),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)
gs.fit(X, y)
best = gs.best_params_
print(f"Mejor configuración: {best}  (AUC interna={gs.best_score_:.4f})")

# 2. CV externa con la mejor configuración + calibración isotónica --------
def factory():
    base = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best)
    # Calibración isotónica con CV 5 dentro del fold de entrenamiento
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

print("\nEjecutando CV 10×5 (RF calibrado)...")
p = cv_predict_proba(factory, X, y)
metrics = evaluate("rf", y, p)
metrics["best_params"] = best

print("\nMétricas (CV 10×5):")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:18s} = {v:.4f}")
    else:
        print(f"  {k:18s} = {v}")

save_metrics_and_preds(RESDIR, "rf", y, p, metrics)
print(f"\nResultados guardados en: {RESDIR}")

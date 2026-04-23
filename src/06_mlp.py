"""
06_mlp.py
---------
MLP (red neuronal) con sklearn. Estandarización dentro del pipeline para
no contaminar folds. Arquitectura pequeña por el tamaño muestral (~900).
"""

from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils_cv import (load_data, cv_predict_proba, evaluate,
                      save_metrics_and_preds, SEED)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados"
X, y, _ = load_data(ROOT)
print(f"n = {len(y)}, casos = {y.sum()} ({y.mean()*100:.1f}%)")

# 1. Tuneo (arquitectura pequeña, regularización L2 alta) -----------------
def make_pipe(**mlp_kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            random_state=SEED,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            **mlp_kwargs,
        ))
    ])

param_grid = {
    "mlp__hidden_layer_sizes": [(8,), (16,), (8, 4), (16, 8)],
    "mlp__alpha":              [1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [1e-3, 1e-2],
}

print("\nGridSearchCV (CV 5-fold, scoring=AUC)...")
gs = GridSearchCV(
    make_pipe(),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)
gs.fit(X, y)
best = gs.best_params_
print(f"Mejor configuración: {best}  (AUC interna={gs.best_score_:.4f})")

# 2. CV externa (calibrado isotónico) -------------------------------------
mlp_kwargs = {k.replace("mlp__", ""): v for k, v in best.items()}

def factory():
    base = make_pipe(**mlp_kwargs)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

print("\nCV 10×5 (MLP calibrado)...")
p = cv_predict_proba(factory, X, y)
metrics = evaluate("mlp", y, p)
metrics["best_params"] = mlp_kwargs

print("\nMétricas (CV 10×5):")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:18s} = {v:.4f}")
    else:
        print(f"  {k:18s} = {v}")

save_metrics_and_preds(RESDIR, "mlp", y, p, metrics)
print(f"\nResultados guardados en: {RESDIR}")

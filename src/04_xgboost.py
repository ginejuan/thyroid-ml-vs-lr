"""
04_xgboost.py
-------------
XGBoost. Tuneo en grid moderado por CV interna 5-fold.
Reportamos versión calibrada (isotónica) y sin calibrar para ver si la
calibración aporta mejora real.
"""

from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils_cv import (load_data, cv_predict_proba, evaluate,
                      save_metrics_and_preds, SEED)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados"
X, y, _ = load_data(ROOT)
print(f"n = {len(y)}, casos = {y.sum()} ({y.mean()*100:.1f}%)")

# 1. Tuneo --------------------------------------------------------------
param_grid = {
    "n_estimators":     [300, 600],
    "max_depth":        [3, 4, 6],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda":       [1.0],
    "min_child_weight": [1, 5],
}
print("\nGridSearchCV (CV 5-fold, scoring=AUC)...")
gs = GridSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=1,            # paralelizamos en GridSearchCV
        tree_method="hist",
        verbosity=0,
    ),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)
gs.fit(X, y)
best = gs.best_params_
print(f"Mejor configuración: {best}  (AUC interna={gs.best_score_:.4f})")

# 2. CV externa (calibrado con isotónica) -------------------------------
def factory():
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        random_state=SEED, n_jobs=1, tree_method="hist",
        verbosity=0, **best
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

print("\nCV 10×5 (XGBoost calibrado)...")
p = cv_predict_proba(factory, X, y)
metrics = evaluate("xgb", y, p)
metrics["best_params"] = best

print("\nMétricas (CV 10×5):")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:18s} = {v:.4f}")
    else:
        print(f"  {k:18s} = {v}")

save_metrics_and_preds(RESDIR, "xgb", y, p, metrics)
print(f"\nResultados guardados en: {RESDIR}")

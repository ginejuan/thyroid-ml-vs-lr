"""
05_bagged_cart.py
-----------------
Bagged CART = BaggingClassifier(DecisionTreeClassifier).
Equivalente al 'treebag' de caret en R.
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from utils_cv import (load_data, cv_predict_proba, evaluate,
                      save_metrics_and_preds, SEED)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados"
X, y, _ = load_data(ROOT)
print(f"n = {len(y)}, casos = {y.sum()} ({y.mean()*100:.1f}%)")

# 1. Tuneo (profundidad/min_samples_leaf del árbol base + n_estimators) ---
def make_bag(n_estimators, max_depth, min_samples_leaf):
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=SEED
        ),
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )

# Pequeño grid manual (BaggingClassifier no expone fácilmente los params del árbol en GridSearchCV)
best_score = -1
best_cfg   = None
print("\nBúsqueda manual de hiperparámetros (CV 5-fold AUC)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for n_est in (200, 500):
    for md in (None, 5, 10):
        for msl in (1, 5, 10):
            aucs = []
            for tr, te in skf.split(X, y):
                m = make_bag(n_est, md, msl)
                m.fit(X[tr], y[tr])
                p_te = m.predict_proba(X[te])[:, 1]
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(y[te], p_te))
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_score:
                best_score = mean_auc
                best_cfg = dict(n_estimators=n_est, max_depth=md,
                                min_samples_leaf=msl)
print(f"Mejor configuración: {best_cfg}  (AUC interna={best_score:.4f})")

# 2. CV externa (calibrado isotónico) -------------------------------------
def factory():
    base = make_bag(**best_cfg)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

print("\nCV 10×5 (Bagged CART calibrado)...")
p = cv_predict_proba(factory, X, y)
metrics = evaluate("bagged_cart", y, p)
metrics["best_params"] = best_cfg

print("\nMétricas (CV 10×5):")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:18s} = {v:.4f}")
    else:
        print(f"  {k:18s} = {v}")

save_metrics_and_preds(RESDIR, "bagged_cart", y, p, metrics)
print(f"\nResultados guardados en: {RESDIR}")

"""
08_temporal_split.py
--------------------
Strict temporal validation:
    - Train: surgeries before 2019 (n=432)
    - Test:  surgeries 2019 onwards (n=429)
For each model: (1) hyperparameter tuning with internal 5-fold CV on the
training cohort only; (2) final fit on the full training cohort; (3)
probability predictions on the test cohort; (4) metrics via
utils_cv.evaluate(). Non-linear models are calibrated with isotonic
regression learned from out-of-fold predictions (CalibratedClassifierCV,
cv=5). Outputs to analisis/resultados_temporal/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from utils_cv import (load_data_temporal, evaluate,
                      save_metrics_and_preds, SEED)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
RESDIR.mkdir(parents=True, exist_ok=True)

X_tr, y_tr, X_te, y_te, predictores = load_data_temporal(ROOT)
print(f"Train: n={len(y_tr)}, casos={y_tr.sum()} ({y_tr.mean()*100:.1f}%)")
print(f"Test : n={len(y_te)}, casos={y_te.sum()} ({y_te.mean()*100:.1f}%)")

INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def fit_eval(name, factory_with_best, best_params):
    """Ajusta en train, evalúa en test, guarda artefactos."""
    print(f"\n--- {name} ---")
    est = factory_with_best()
    est.fit(X_tr, y_tr)
    p_te = est.predict_proba(X_te)[:, 1]
    metrics = evaluate(name, y_te, p_te)
    metrics["best_params"] = best_params
    metrics["n_train"]     = int(len(y_tr))
    metrics["n_test"]      = int(len(y_te))
    print("Métricas en test (cohorte 2019+):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s} = {v:.4f}")
    save_metrics_and_preds(RESDIR, name, y_te, p_te, metrics)


# =====================================================================
# 1. LOGISTIC REGRESSION (sin tuneo)
# =====================================================================
def make_logistic():
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)

# Coeficientes finales en train (informativo)
mod = make_logistic().fit(X_tr, y_tr)
coef_tbl = pd.DataFrame({
    "variable": ["(Intercept)"] + predictores,
    "estimate": [mod.intercept_[0]] + mod.coef_[0].tolist(),
    "OR":       [np.exp(mod.intercept_[0])] + np.exp(mod.coef_[0]).tolist()
})
print("\nCoeficientes regresión logística (entrenada en train pre-2019):")
print(coef_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
coef_tbl.to_csv(RESDIR / "logistic_coefs_train.csv", index=False)

fit_eval("logistic", make_logistic, best_params={})


# =====================================================================
# 2. RANDOM FOREST (tuning interno)
# =====================================================================
print("\n[RF] GridSearchCV sobre train...")
rf_grid = {
    "n_estimators":     [500],
    "max_features":     [3, 4, 5, 6, "sqrt"],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_depth":        [None],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_grid=rf_grid, cv=INNER, scoring="roc_auc", n_jobs=-1, refit=True,
)
gs.fit(X_tr, y_tr)
rf_best = gs.best_params_
print(f"RF best: {rf_best}  AUC interna train={gs.best_score_:.4f}")

def make_rf():
    base = RandomForestClassifier(random_state=SEED, n_jobs=-1, **rf_best)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

fit_eval("rf", make_rf, rf_best)


# =====================================================================
# 3. XGBOOST (tuning interno)
# =====================================================================
print("\n[XGB] GridSearchCV sobre train...")
xgb_grid = {
    "n_estimators":     [300, 600],
    "max_depth":        [3, 4, 6],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda":       [1.0],
    "min_child_weight": [1, 5],
}
gs = GridSearchCV(
    XGBClassifier(objective="binary:logistic", eval_metric="auc",
                  random_state=SEED, n_jobs=1, tree_method="hist", verbosity=0),
    param_grid=xgb_grid, cv=INNER, scoring="roc_auc", n_jobs=-1, refit=True,
)
gs.fit(X_tr, y_tr)
xgb_best = gs.best_params_
print(f"XGB best: {xgb_best}  AUC interna train={gs.best_score_:.4f}")

def make_xgb():
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        random_state=SEED, n_jobs=1, tree_method="hist",
        verbosity=0, **xgb_best
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

fit_eval("xgb", make_xgb, xgb_best)


# =====================================================================
# 4. BAGGED CART (búsqueda manual sobre train)
# =====================================================================
print("\n[BaggedCART] búsqueda manual sobre train (CV 5-fold)...")
def make_bag(n_estimators, max_depth, min_samples_leaf):
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=SEED),
        n_estimators=n_estimators, bootstrap=True,
        random_state=SEED, n_jobs=-1,
    )

best_score = -1; best_cfg = None
for n_est in (200, 500):
    for md in (None, 5, 10):
        for msl in (1, 5, 10):
            aucs = []
            for tr, te in INNER.split(X_tr, y_tr):
                m = make_bag(n_est, md, msl)
                m.fit(X_tr[tr], y_tr[tr])
                aucs.append(roc_auc_score(y_tr[te],
                                          m.predict_proba(X_tr[te])[:, 1]))
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_score:
                best_score = mean_auc
                best_cfg = dict(n_estimators=n_est, max_depth=md,
                                min_samples_leaf=msl)
print(f"BaggedCART best: {best_cfg}  AUC interna train={best_score:.4f}")

def make_bagged():
    base = make_bag(**best_cfg)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

fit_eval("bagged_cart", make_bagged, best_cfg)


# =====================================================================
# 5. MLP (tuning interno)
# =====================================================================
print("\n[MLP] GridSearchCV sobre train...")
def make_mlp_pipe(**kw):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            random_state=SEED, max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15, **kw))
    ])

mlp_grid = {
    "mlp__hidden_layer_sizes": [(8,), (16,), (8, 4), (16, 8)],
    "mlp__alpha":              [1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [1e-3, 1e-2],
}
gs = GridSearchCV(make_mlp_pipe(), param_grid=mlp_grid,
                  cv=INNER, scoring="roc_auc", n_jobs=-1, refit=True)
gs.fit(X_tr, y_tr)
mlp_best = {k.replace("mlp__", ""): v for k, v in gs.best_params_.items()}
print(f"MLP best: {mlp_best}  AUC interna train={gs.best_score_:.4f}")

def make_mlp():
    base = make_mlp_pipe(**mlp_best)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)

fit_eval("mlp", make_mlp, mlp_best)

print(f"\n=== Listo. Resultados en {RESDIR} ===")

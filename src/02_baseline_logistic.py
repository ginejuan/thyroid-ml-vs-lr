"""
02_baseline_logistic.py
-----------------------
Baseline: logistic regression with the published predictor specification.
Uses the shared cross-validation scheme (utils_cv.py) so that metrics are
directly comparable with the ML models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils_cv import (load_data, cv_predict_proba, evaluate,
                      save_metrics_and_preds)

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados"

X, y, predictores = load_data(ROOT)
print(f"n = {len(y)}, casos = {y.sum()} ({y.mean()*100:.1f}%)")

# 1. Modelo en sample completo: solo para reportar coeficientes/OR --------
mod = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
mod.fit(X, y)
coef_tbl = pd.DataFrame({
    "variable": ["(Intercept)"] + predictores,
    "estimate": [mod.intercept_[0]] + mod.coef_[0].tolist(),
    "OR":       [np.exp(mod.intercept_[0])] + np.exp(mod.coef_[0]).tolist()
})
print("\nCoeficientes (regresión logística re-ajustada en cohorte n=888):")
print(coef_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
coef_tbl.to_csv(RESDIR / "baseline_logistic_coefs.csv", index=False)

# 2. Predicciones por CV y métricas estándar ------------------------------
def factory():
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)

print("\nEjecutando CV 10×5...")
p = cv_predict_proba(factory, X, y)
metrics = evaluate("logistic", y, p)

print("\nMétricas (CV 10×5):")
for k, v in metrics.items():
    print(f"  {k:18s} = {v:.4f}" if isinstance(v, float) else f"  {k:18s} = {v}")

save_metrics_and_preds(RESDIR, "logistic", y, p, metrics)
print(f"\nResultados guardados en: {RESDIR}")

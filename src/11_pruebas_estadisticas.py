"""
11_pruebas_estadisticas.py
--------------------------
Formal statistical tests: (1) paired DeLong test of each ML model vs the
logistic-regression baseline on the temporal test cohort; (2) paired
bootstrap (2,000 replicates) for Brier-score differences; (3) paired
bootstrap for calibration-slope differences.
Output: analisis/resultados_temporal/pruebas_estadisticas.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
SEED   = 20260422

# ------------------------------------------------------------------
# Helper: Test de DeLong (Sun & Xu 2014, implementación numpy pura)
# ------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Implementa el cálculo rápido de DeLong (Sun & Xu, IEEE SPL 2014).
    predictions_sorted_transposed: array (k_classifiers, N), N=N_pos+N_neg
        Las primeras label_1_count columnas son positivos.
    Devuelve aucs (k,) y la matriz de covarianzas (k,k).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(positive_examples[r, :])
        ty[r, :] = _midrank(negative_examples[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, p1, p2):
    """
    Test de DeLong para AUCs pareadas. Devuelve dict con
    auc1, auc2, diff, var, z, p (dos colas).
    """
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack([p1[order], p2[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0:
        z = 0.0
        pval = 1.0
    else:
        z = diff / np.sqrt(var)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "diff": float(diff),
        "se":   float(np.sqrt(max(var, 0))),
        "z":    float(z),
        "p":    float(pval),
    }


# ------------------------------------------------------------------
# Helper: bootstrap pareado para diferencia de métricas
# ------------------------------------------------------------------
def _cal_slope(y, p):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    m.fit(lp.reshape(-1, 1), y)
    return float(m.coef_[0][0])


def paired_bootstrap_diff(y, p1, p2, metric_fn,
                          n_boot: int = 2000, seed: int = SEED):
    """
    Bootstrap percentil estratificado pareado: en cada réplica se
    muestrean los MISMOS índices para ambos modelos (capturando la
    correlación entre predicciones).
    Devuelve dict con metric1, metric2, diff, ci95, p (dos colas
    aproximada por inversión del IC bootstrap).
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ip   = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        idx  = np.concatenate([ip, ineg])
        diffs[b] = metric_fn(y[idx], p1[idx]) - metric_fn(y[idx], p2[idx])
    m1 = metric_fn(y, p1)
    m2 = metric_fn(y, p2)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p-valor aproximado: 2 * min(P(diff>=0), P(diff<=0))
    p_left  = (diffs <= 0).mean()
    p_right = (diffs >= 0).mean()
    pval = 2 * min(p_left, p_right)
    return {
        "metric1": float(m1),
        "metric2": float(m2),
        "diff":    float(m1 - m2),
        "ci_lo":   float(lo),
        "ci_hi":   float(hi),
        "p":       float(min(pval, 1.0)),
    }


# ------------------------------------------------------------------
# Carga de predicciones por modelo (test cohort)
# ------------------------------------------------------------------
def load_pred(name):
    df = pd.read_csv(RESDIR / f"{name}_predictions.csv")
    return df["y"].astype(int).values, df[f"p_{name}"].values


y_l, p_log     = load_pred("logistic")
_,  p_rf       = load_pred("rf")
_,  p_xgb      = load_pred("xgb")
_,  p_bag      = load_pred("bagged_cart")
_,  p_mlp      = load_pred("mlp")

# Comprobación: misma y para todos
for name, p in [("rf", p_rf), ("xgb", p_xgb), ("bagged_cart", p_bag), ("mlp", p_mlp)]:
    yi, _ = load_pred(name)
    assert np.array_equal(yi, y_l), f"y diferente para {name}!"

models = [("rf", "Random Forest", p_rf),
          ("xgb", "XGBoost", p_xgb),
          ("bagged_cart", "Bagged CART", p_bag),
          ("mlp", "MLP", p_mlp)]

rows = []
print("=== Tests pareados vs. logistic regression (test cohort, n=429) ===\n")
for name, label, p_ml in models:
    # 1. DeLong AUC
    d  = delong_test(y_l, p_log, p_ml)
    # 2. Bootstrap Brier
    bb = paired_bootstrap_diff(y_l, p_log, p_ml, brier_score_loss)
    # 3. Bootstrap calibration slope
    bs = paired_bootstrap_diff(y_l, p_log, p_ml, _cal_slope)

    rows.append({
        "comparison":         f"Logistic vs. {label}",
        "AUC_log":            round(d["auc1"], 3),
        "AUC_ml":             round(d["auc2"], 3),
        "AUC_diff":           round(d["diff"], 3),
        "AUC_p_DeLong":       round(d["p"], 3),
        "Brier_log":          round(bb["metric1"], 4),
        "Brier_ml":           round(bb["metric2"], 4),
        "Brier_diff":         round(bb["diff"], 4),
        "Brier_p_bootstrap":  round(bb["p"], 3),
        "Slope_log":          round(bs["metric1"], 2),
        "Slope_ml":           round(bs["metric2"], 2),
        "Slope_diff":         round(bs["diff"], 2),
        "Slope_p_bootstrap":  round(bs["p"], 3),
    })
    print(f"--- Logistic vs. {label} ---")
    print(f"  AUC:   log={d['auc1']:.3f}  ml={d['auc2']:.3f}  "
          f"diff={d['diff']:+.3f}  p(DeLong)={d['p']:.3f}")
    print(f"  Brier: log={bb['metric1']:.4f}  ml={bb['metric2']:.4f}  "
          f"diff={bb['diff']:+.4f}  p(boot)={bb['p']:.3f}")
    print(f"  Slope: log={bs['metric1']:.2f}  ml={bs['metric2']:.2f}  "
          f"diff={bs['diff']:+.2f}  p(boot)={bs['p']:.3f}\n")

tabla = pd.DataFrame(rows)
out = RESDIR / "pruebas_estadisticas.csv"
tabla.to_csv(out, index=False)
print(f"Guardado: {out}")

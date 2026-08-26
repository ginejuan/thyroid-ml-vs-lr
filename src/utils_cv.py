"""
utils_cv.py
-----------
Shared cross-validation scheme and metrics for all models. Guarantees
that every model is evaluated on EXACTLY the same folds (same seed, same
number of splits and repeats) and reported with the same metrics for an
honest comparison.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from scipy.special import logit

SEED       = 20260422
N_FOLDS    = 10
N_REPEATS  = 5
CUTOFF_PAPER = 0.0955  # cutoff óptimo del paper de validación


def cv_predict_proba(estimator_factory, X, y,
                     n_folds: int = N_FOLDS,
                     n_repeats: int = N_REPEATS,
                     seed: int = SEED) -> np.ndarray:
    """
    Predicciones por CV estratificada repetida. Promedia probabilidades a
    través de los repeats por paciente.

    estimator_factory: callable que devuelve un estimador NUEVO cada vez
                       (evita estado entre folds).
    """
    proba_sum   = np.zeros(len(y))
    proba_count = np.zeros(len(y))

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + repeat)
        for tr, te in skf.split(X, y):
            est = estimator_factory()
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[te])[:, 1]
            proba_sum[te]   += p
            proba_count[te] += 1

    assert (proba_count == n_repeats).all(), \
        "Algún paciente no fue predicho en todos los repeats"
    return proba_sum / proba_count


def auc_with_ci(y, p, n_boot: int = 2000, seed: int = SEED):
    """AUC con IC95% por bootstrap percentil estratificado."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        aucs[b] = roc_auc_score(y[idx], p[idx])
    auc = roc_auc_score(y, p)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def calibration_slope_intercept(y, p):
    """
    Calibration slope e intercept: glm(y ~ logit(p)).
    Slope=1, intercept=0 es la calibración ideal.
    """
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    m.fit(lp.reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def cutoff_metrics(y, p, cutoff: float):
    """Sensibilidad y especificidad a un cutoff dado."""
    pred = (p >= cutoff).astype(int)
    sens = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return sens, spec


def youden_cutoff(y, p):
    """Devuelve cutoff Youden, sens y spec en ese cutoff."""
    fpr, tpr, thr = roc_curve(y, p)
    j = (tpr - fpr).argmax()
    return float(thr[j]), float(tpr[j]), float(1 - fpr[j])


def evaluate(name: str, y, p, n_boot: int = 2000) -> dict:
    """
    Construye el dict estándar de métricas para un modelo.
    """
    auc, auc_lo, auc_hi = auc_with_ci(y, p, n_boot=n_boot)
    brier = float(brier_score_loss(y, p))
    cs, ci = calibration_slope_intercept(y, p)
    co_y, sens_y, spec_y = youden_cutoff(y, p)
    sens_p, spec_p = cutoff_metrics(y, p, CUTOFF_PAPER)
    return {
        "model":          name,
        "n":              int(len(y)),
        "auc":            auc,
        "auc_lo":         auc_lo,
        "auc_hi":         auc_hi,
        "brier":          brier,
        "cal_slope":      cs,
        "cal_intercept":  ci,
        "cutoff_youden":  co_y,
        "sens_youden":    sens_y,
        "spec_youden":    spec_y,
        "cutoff_paper":   CUTOFF_PAPER,
        "sens_paper":     sens_p,
        "spec_paper":     spec_p,
    }


def save_metrics_and_preds(out_dir: Path, name: str, y, p, metrics: dict):
    """Guarda métricas (JSON) y predicciones (CSV) por modelo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"y": y, f"p_{name}": p}).to_csv(
        out_dir / f"{name}_predictions.csv", index=False)


PREDICTORES = [
    "fhx_tc", "male", "age", "age2",
    "tsh_low", "tsh_high",
    "autoimmune", "solid", "susp_nodes", "hypoechoic",
    "irregular", "macrocalc", "microcalc", "taller_than_wide"
]


def load_data(root: Path):
    """Carga el dataset complete-case y devuelve (X, y, predictores)."""
    dat = pd.read_csv(root / "analisis" / "datos_modelo_cc.csv")
    X = dat[PREDICTORES].values.astype(float)
    y = dat["cancer"].astype(int).values
    return X, y, PREDICTORES


def load_data_temporal(root: Path):
    """
    Carga el dataset con split temporal y devuelve
    (X_train, y_train, X_test, y_test, predictores).
    """
    dat = pd.read_csv(root / "analisis" / "datos_modelo_cc_temporal.csv")
    train = dat[dat["cohort"] == "train"]
    test  = dat[dat["cohort"] == "test"]
    X_tr = train[PREDICTORES].values.astype(float)
    y_tr = train["cancer"].astype(int).values
    X_te = test[PREDICTORES].values.astype(float)
    y_te = test["cancer"].astype(int).values
    return X_tr, y_tr, X_te, y_te, PREDICTORES

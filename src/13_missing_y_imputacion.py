"""
13_missing_y_imputacion.py
--------------------------
Missing-data analysis and multiple-imputation sensitivity analysis:
(A) missingness per variable and association of missingness patterns
with the outcome; (B) MICE (m=20, IterativeImputer with BayesianRidge,
sample_posterior=True): models re-trained on each imputed training set,
evaluated on the imputed test set, predictions pooled across imputations
(Rubin's rule for prediction).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
SEED   = 20260422

# Dataset COMPLETO (con NAs) y dataset complete-case ya separado por cohorte
dat_full = pd.read_csv(ROOT / "analisis" / "datos_modelo.csv")
dat_full = dat_full.dropna(subset=["cancer", "year_surgery"]).copy()
dat_full["year_surgery"] = dat_full["year_surgery"].astype(int)

predictors = [
    "fhx_tc", "male", "age", "age2",
    "tsh_low", "tsh_high",
    "autoimmune", "solid", "susp_nodes", "hypoechoic",
    "irregular", "macrocalc", "microcalc", "taller_than_wide"
]

n_total = len(dat_full)
print(f"Cohorte con outcome y año de cirugía: n={n_total}")

# ---------------------------------------------------------------------
# A. Patrones de missingness
# ---------------------------------------------------------------------
print("\n=== A. Missingness por variable ===")
miss = dat_full[predictors].isna().sum().sort_values(ascending=False)
miss_pct = (miss / n_total * 100).round(2)
miss_tbl = pd.DataFrame({
    "variable":   miss.index,
    "n_missing":  miss.values,
    "percent":    miss_pct.values,
})

# Asociación entre estar missing en la variable y el outcome (chi2)
out = dat_full["cancer"].astype(int).values
mcar_rows = []
for v in predictors:
    is_miss = dat_full[v].isna().astype(int).values
    if is_miss.sum() == 0:
        mcar_rows.append({"variable": v, "n_missing": 0, "p_vs_outcome": np.nan})
        continue
    table = np.array([
        [((is_miss == 1) & (out == 1)).sum(),
         ((is_miss == 1) & (out == 0)).sum()],
        [((is_miss == 0) & (out == 1)).sum(),
         ((is_miss == 0) & (out == 0)).sum()],
    ])
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        p = np.nan
    else:
        p = stats.chi2_contingency(table, correction=False).pvalue
    mcar_rows.append({"variable": v, "n_missing": int(is_miss.sum()),
                      "p_vs_outcome": p})
mcar_df = pd.DataFrame(mcar_rows)
miss_tbl = miss_tbl.merge(mcar_df[["variable", "p_vs_outcome"]], on="variable")
print(miss_tbl.to_string(index=False))
miss_tbl.to_csv(RESDIR / "missingness_summary.csv", index=False)

n_complete = dat_full[predictors].dropna().shape[0]
print(f"\nCasos completos en los 13 predictores: {n_complete}/{n_total} "
      f"({n_complete/n_total*100:.1f}%)")
print(f"Pacientes excluidos por missing: {n_total - n_complete}")

# ---------------------------------------------------------------------
# B. Imputación múltiple (MICE) y re-entrenamiento de la LR
# ---------------------------------------------------------------------
print("\n=== B. Imputación múltiple (MICE, m=20) sobre cohorte temporal ===")
m = 20
train = dat_full[dat_full["year_surgery"] < 2019].copy()
test  = dat_full[dat_full["year_surgery"] >= 2019].copy()
y_tr_full = train["cancer"].astype(int).values
y_te_full = test["cancer"].astype(int).values
print(f"Train (con missing): n={len(train)}  |  Test: n={len(test)}")

aucs, briers = [], []
preds_test_acc = np.zeros(len(test))

for i in range(m):
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20, sample_posterior=True,
        random_state=SEED + i, initial_strategy="median"
    )
    Xtr = imp.fit_transform(train[predictors])
    Xte = imp.transform(test[predictors])

    # Asegurar binarización de variables binarias tras imputación
    binary_cols = [predictors.index(c) for c in predictors
                   if c not in ("age", "age2")]
    Xtr_bin = Xtr.copy(); Xte_bin = Xte.copy()
    Xtr_bin[:, binary_cols] = (Xtr_bin[:, binary_cols] >= 0.5).astype(float)
    Xte_bin[:, binary_cols] = (Xte_bin[:, binary_cols] >= 0.5).astype(float)
    # Recalcular age2 a partir de age
    Xtr_bin[:, predictors.index("age2")] = Xtr_bin[:, predictors.index("age")] ** 2
    Xte_bin[:, predictors.index("age2")] = Xte_bin[:, predictors.index("age")] ** 2

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(Xtr_bin, y_tr_full)
    p = lr.predict_proba(Xte_bin)[:, 1]
    preds_test_acc += p
    aucs.append(roc_auc_score(y_te_full, p))
    briers.append(brier_score_loss(y_te_full, p))

preds_test_acc /= m
auc_pooled   = float(roc_auc_score(y_te_full, preds_test_acc))
brier_pooled = float(brier_score_loss(y_te_full, preds_test_acc))

print(f"AUC promedio individual:  {np.mean(aucs):.3f}  "
      f"(rango {np.min(aucs):.3f}–{np.max(aucs):.3f})")
print(f"Brier promedio individual: {np.mean(briers):.4f}  "
      f"(rango {np.min(briers):.4f}–{np.max(briers):.4f})")
print(f"AUC con probabilidades agrupadas (Rubin):   {auc_pooled:.3f}")
print(f"Brier con probabilidades agrupadas (Rubin): {brier_pooled:.4f}")

# Comparación con complete-case (modelo logístico publicado, ya guardado)
cc_metrics_path = RESDIR / "logistic_metrics.json"
import json
cc = json.load(open(cc_metrics_path))
print(f"\nReferencia complete-case:")
print(f"  AUC   = {cc['auc']:.3f}  (95% IC {cc['auc_lo']:.3f}–{cc['auc_hi']:.3f})")
print(f"  Brier = {cc['brier']:.4f}")

result = pd.DataFrame([{
    "Analysis":           "Complete-case",
    "n_train":            cc.get("n_train", "—"),
    "n_test":             cc.get("n", "—"),
    "AUC":                round(cc["auc"], 3),
    "AUC_95CI":           f"({cc['auc_lo']:.3f}–{cc['auc_hi']:.3f})",
    "Brier":              round(cc["brier"], 4),
}, {
    "Analysis":           f"Multiple imputation (MICE, m={m})",
    "n_train":            int(len(train)),
    "n_test":             int(len(test)),
    "AUC":                round(auc_pooled, 3),
    "AUC_95CI":           f"individual range {np.min(aucs):.3f}–{np.max(aucs):.3f}",
    "Brier":              round(brier_pooled, 4),
}])
print("\n=== Sensibilidad: complete-case vs. imputación múltiple ===")
print(result.to_string(index=False))
out_path = RESDIR / "imputacion_multiple_resultado.csv"
result.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}")

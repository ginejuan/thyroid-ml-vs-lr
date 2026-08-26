"""
12_linealidad_interacciones.py
------------------------------
Model-specification checks for the logistic regression: (A) linearity of
age via likelihood-ratio test comparing the paper specification
(age + age^2) against restricted cubic splines; (B) pre-specified
two-way interactions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from patsy import dmatrix

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"

dat   = pd.read_csv(ROOT / "analisis" / "datos_modelo_cc_temporal.csv")
train = dat[dat["cohort"] == "train"].copy().reset_index(drop=True)

base_predictors = [
    "fhx_tc", "male", "tsh_low", "tsh_high",
    "autoimmune", "solid", "susp_nodes", "hypoechoic",
    "irregular", "macrocalc", "microcalc", "taller_than_wide"
]

y = train["cancer"].astype(int).values

def fit_logit(X):
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    return model.fit(disp=0, maxiter=200)

def lrt(res_full, res_reduced):
    df = res_full.df_model - res_reduced.df_model
    stat = 2 * (res_full.llf - res_reduced.llf)
    p = 1 - stats.chi2.cdf(stat, df)
    return float(stat), int(df), float(p)


# =================================================================
# A. LINEALIDAD DE LA EDAD
# =================================================================
print("=== A. Linealidad de la edad (RCS, 4 nodos) ===")
X_lin = train[base_predictors + ["age", "age2"]].astype(float).values
res_lin = fit_logit(X_lin)
print(f"Modelo lineal+cuadrático: log-lik = {res_lin.llf:.2f}, AIC = {res_lin.aic:.2f}")

# RCS con 4 nodos en cuantiles 5%, 35%, 65%, 95% (recomendación de Harrell)
knots = np.percentile(train["age"].values, [5, 35, 65, 95])
formula = f"cr(age, knots={list(knots)})"
rcs_basis = dmatrix(formula, train, return_type="dataframe").iloc[:, 1:]  # quita intercepto interno
rcs_basis.columns = [f"age_rcs{i+1}" for i in range(rcs_basis.shape[1])]
X_rcs = pd.concat([train[base_predictors].astype(float).reset_index(drop=True),
                   rcs_basis.reset_index(drop=True)], axis=1).values
res_rcs = fit_logit(X_rcs)
print(f"Modelo con RCS:           log-lik = {res_rcs.llf:.2f}, AIC = {res_rcs.aic:.2f}")

# LRT (RCS engloba al lineal: 3 grados de libertad extra para RCS, 2 para age+age2)
stat_age, df_age, p_age = lrt(res_rcs, res_lin)
print(f"LRT (RCS vs. age+age2):  chi2 = {stat_age:.2f}, df = {df_age}, p = {p_age:.3f}")

linealidad_row = {
    "test": "Linearity of age (RCS, 4 knots vs. age + age2)",
    "chi2": round(stat_age, 2),
    "df":   df_age,
    "p":    round(p_age, 3),
    "interpretation": "linear specification adequate" if p_age > 0.05
                      else "evidence of residual non-linearity",
}

# =================================================================
# B. INTERACCIONES DE DOS VÍAS
# =================================================================
print("\n=== B. Interacciones de dos vías ===")
# Modelo base (especificación del paper original)
X_base = train[base_predictors + ["age", "age2"]].astype(float).values
res_base = fit_logit(X_base)
print(f"Modelo base: log-lik = {res_base.llf:.2f}, AIC = {res_base.aic:.2f}\n")

interactions = [
    ("age", "male"),
    ("age", "fhx_tc"),
    ("microcalc", "hypoechoic"),
    ("microcalc", "irregular"),
    ("solid", "hypoechoic"),
    ("solid", "tsh_low"),
    ("tsh_low", "fhx_tc"),
    ("autoimmune", "tsh_high"),
    ("susp_nodes", "irregular"),
    ("taller_than_wide", "microcalc"),
]

inter_rows = []
n_tests = len(interactions)
for v1, v2 in interactions:
    inter_term = (train[v1].astype(float).values * train[v2].astype(float).values
                  ).reshape(-1, 1)
    X_inter = np.hstack([X_base, inter_term])
    res_inter = fit_logit(X_inter)
    stat, df, p = lrt(res_inter, res_base)
    p_bonf = min(p * n_tests, 1.0)
    inter_rows.append({
        "interaction":  f"{v1} x {v2}",
        "chi2":         round(stat, 2),
        "df":           df,
        "p":            round(p, 3),
        "p_Bonferroni": round(p_bonf, 3),
        "significant_unadj": p < 0.05,
    })
    print(f"  {v1:20s} x {v2:20s}  chi2={stat:5.2f}  p={p:.3f}  p_bonf={p_bonf:.3f}")

# =================================================================
# Guardar resultados
# =================================================================
df_lin   = pd.DataFrame([linealidad_row])
df_inter = pd.DataFrame(inter_rows)

out_lin   = RESDIR / "linealidad_edad.csv"
out_inter = RESDIR / "interacciones_dos_vias.csv"
df_lin.to_csv(out_lin, index=False)
df_inter.to_csv(out_inter, index=False)
print(f"\nGuardado: {out_lin}")
print(f"Guardado: {out_inter}")

n_sig_unadj = int(df_inter["significant_unadj"].sum())
n_sig_bonf  = int((df_inter["p_Bonferroni"] < 0.05).sum())
print(f"\nResumen interacciones: {n_sig_unadj}/{n_tests} sin ajustar, "
      f"{n_sig_bonf}/{n_tests} tras Bonferroni.")

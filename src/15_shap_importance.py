"""
15_shap_importance.py
---------------------
SHAP variable importance for the best-performing ML models (Random
Forest and XGBoost) on the temporal test cohort, compared descriptively
with the standardised logistic-regression coefficients. Broadly similar
global rankings support the conclusion that the ML models are not
exploiting hidden interactions or non-linearities missed by the LR.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import RandomForestClassifier
from xgboost import XGBClassifier

from utils_cv import load_data_temporal, SEED

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
FIGDIR = ROOT / "analisis" / "figuras_temporal"
RESDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

X_tr, y_tr, X_te, y_te, predictores = load_data_temporal(ROOT)
X_tr_df = pd.DataFrame(X_tr, columns=predictores)
X_te_df = pd.DataFrame(X_te, columns=predictores)
print(f"Train n={len(y_tr)}  Test n={len(y_te)}  predictores={len(predictores)}")

# Etiquetas legibles para las figuras
LABELS = {
    "fhx_tc":           "Antecedente familiar CDT",
    "male":             "Sexo masculino",
    "age":              "Edad (lineal)",
    "age2":             "Edad (cuadrática)",
    "tsh_low":          "TSH < 0,369",
    "tsh_high":         "TSH > 4,701",
    "autoimmune":       "Tiroiditis autoinmune",
    "solid":            "Nódulo sólido",
    "susp_nodes":       "Adenopatías sospechosas",
    "hypoechoic":       "Nódulo hipoecoico",
    "irregular":        "Márgenes irregulares",
    "macrocalc":        "Macrocalcificaciones",
    "microcalc":        "Microcalcificaciones",
    "taller_than_wide": "Más alto que ancho",
}
labels_pretty = [LABELS[v] for v in predictores]

# =====================================================================
# 1. Refitting de RF y XGB con los mejores hiperparámetros (sin
#    calibración, porque queremos la importancia del modelo subyacente,
#    no del wrapper isotónico)
# =====================================================================
rf_best  = json.load(open(RESDIR / "rf_metrics.json"))["best_params"]
xgb_best = json.load(open(RESDIR / "xgb_metrics.json"))["best_params"]

rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **rf_best)
rf.fit(X_tr_df, y_tr)

xgb = XGBClassifier(
    objective="binary:logistic", eval_metric="auc",
    random_state=SEED, n_jobs=1, tree_method="hist",
    verbosity=0, base_score=0.5, **xgb_best,
)
xgb.fit(X_tr_df, y_tr)

# =====================================================================
# 2. SHAP (TreeExplainer en ambos)
# =====================================================================
print("[SHAP] RF ...")
expl_rf  = shap.TreeExplainer(rf)
sv_rf    = expl_rf.shap_values(X_te_df, check_additivity=False)
# RandomForestClassifier devuelve lista (clase 0, clase 1) o matriz
# tridimensional (n, p, 2) según versión de SHAP. Nos quedamos con clase 1.
if isinstance(sv_rf, list):
    sv_rf = sv_rf[1]
elif sv_rf.ndim == 3:
    sv_rf = sv_rf[:, :, 1]

print("[SHAP] XGB (vía pred_contribs nativo de XGBoost) ...")
# El TreeExplainer de SHAP es incompatible con la nueva serialización JSON
# de XGBoost (base_score como '[5E-1]'). Usamos el cálculo SHAP nativo de
# XGBoost, que devuelve valores idénticos.
import xgboost as _xgb_mod
_dtest    = _xgb_mod.DMatrix(X_te_df.values, feature_names=predictores)
_contribs = xgb.get_booster().predict(_dtest, pred_contribs=True)
# Última columna es el bias; nos quedamos con las p columnas de variables
sv_xgb = _contribs[:, :-1]

mean_abs_rf  = np.abs(sv_rf).mean(axis=0)
mean_abs_xgb = np.abs(sv_xgb).mean(axis=0)

# =====================================================================
# 3. Coeficientes estandarizados de la RL ajustada en train
#    (refitting con los mismos predictores; equivalente al baseline
#     publicado en el artículo de Diagnostics 2025 sobre la cohorte
#     temporal-train)
# =====================================================================
sd_train = X_tr_df.std(ddof=1).values
lr = LogisticRegression(penalty=None, max_iter=2000, solver="lbfgs")
lr.fit(X_tr_df, y_tr)
beta = lr.coef_.ravel()
std_coef = np.abs(beta * sd_train)        # |β_std| comparable entre vars

# =====================================================================
# 4. Ranking comparado
# =====================================================================
ranking = pd.DataFrame({
    "variable":         predictores,
    "etiqueta":         labels_pretty,
    "shap_rf":          mean_abs_rf,
    "shap_xgb":         mean_abs_xgb,
    "lr_std_coef_abs":  std_coef,
    "lr_OR":            np.exp(beta),
}).sort_values("shap_xgb", ascending=False)

# Rangos (1 = más importante)
ranking["rank_rf"]  = ranking["shap_rf"].rank(ascending=False).astype(int)
ranking["rank_xgb"] = ranking["shap_xgb"].rank(ascending=False).astype(int)
ranking["rank_lr"]  = ranking["lr_std_coef_abs"].rank(ascending=False).astype(int)
ranking.to_csv(RESDIR / "shap_importance_ranking.csv", index=False)
print(ranking[["etiqueta", "rank_lr", "rank_rf", "rank_xgb"]].to_string(index=False))

# Concordancia de ranking
from scipy.stats import spearmanr
rho_rf,  p_rf  = spearmanr(ranking["lr_std_coef_abs"], ranking["shap_rf"])
rho_xgb, p_xgb = spearmanr(ranking["lr_std_coef_abs"], ranking["shap_xgb"])
print(f"Spearman LR vs RF : rho={rho_rf:.3f}  (p={p_rf:.3f})")
print(f"Spearman LR vs XGB: rho={rho_xgb:.3f}  (p={p_xgb:.3f})")

with open(RESDIR / "shap_concordancia.txt", "w") as f:
    f.write(f"Spearman LR vs RF : rho={rho_rf:.3f}  (p={p_rf:.3f})\n")
    f.write(f"Spearman LR vs XGB: rho={rho_xgb:.3f}  (p={p_xgb:.3f})\n")

# =====================================================================
# 5. Figura SHAP summary (barras) — paneles RF y XGB
# =====================================================================
order = np.argsort(mean_abs_xgb)        # ascendente (de menor a mayor)
y_pos = np.arange(len(order))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), sharey=True)
for ax, vals, title in (
    (axes[0], mean_abs_rf,  "Random Forest"),
    (axes[1], mean_abs_xgb, "XGBoost"),
):
    ax.barh(y_pos, vals[order], color="#2c7fb8", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels_pretty[i] for i in order], fontsize=9)
    ax.set_xlabel("|SHAP| medio (test temporal, n=429)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)

plt.suptitle(
    "Figura S1. Importancia de variables (SHAP) en los dos mejores modelos ML",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(FIGDIR / "figura_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"-> figura_shap_summary.png")

# =====================================================================
# 6. Figura SHAP beeswarm (XGBoost) — distribución de efectos
# =====================================================================
plt.figure(figsize=(8.5, 6.5))
shap.summary_plot(
    sv_xgb, X_te_df,
    feature_names=labels_pretty,
    plot_type="dot",
    show=False,
    color_bar_label="Valor de la variable",
)
plt.title(
    "Figura S2. Distribución de valores SHAP (XGBoost, cohorte test)",
    fontsize=11, fontweight="bold", pad=14,
)
plt.tight_layout()
plt.savefig(FIGDIR / "figura_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"-> figura_shap_beeswarm.png")

print("\n=== Listo ===")

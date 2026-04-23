"""
09_comparar_temporal.py
-----------------------
Equivalente a 07_comparar_modelos.py pero leyendo
analisis/resultados_temporal/ (modelos entrenados en pre-2019,
evaluados en cohorte 2019+).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
FIGDIR = ROOT / "analisis" / "figuras_temporal"
FIGDIR.mkdir(exist_ok=True)

MODELS = [
    ("logistic",    "Regresión logística"),
    ("rf",          "Random Forest"),
    ("xgb",         "XGBoost"),
    ("bagged_cart", "Bagged CART"),
    ("mlp",         "MLP"),
]
COLORS = dict(zip([m[0] for m in MODELS],
                  ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]))

# 1. Cargar -------------------------------------------------------------
metrics_list = []
preds = {}
for key, _ in MODELS:
    with open(RESDIR / f"{key}_metrics.json") as f:
        metrics_list.append(json.load(f))
    df_p = pd.read_csv(RESDIR / f"{key}_predictions.csv")
    preds[key] = (df_p["y"].values, df_p[f"p_{key}"].values)

# 2. Tabla --------------------------------------------------------------
tbl = pd.DataFrame(metrics_list)[
    ["model", "n", "auc", "auc_lo", "auc_hi", "brier",
     "cal_slope", "cal_intercept",
     "cutoff_youden", "sens_youden", "spec_youden",
     "sens_paper", "spec_paper"]
]
tbl["AUC (IC95%)"] = tbl.apply(
    lambda r: f"{r.auc:.3f} ({r.auc_lo:.3f}–{r.auc_hi:.3f})", axis=1)

print("\n=== TABLA COMPARATIVA — VALIDACIÓN TEMPORAL (test 2019+) ===\n")
print(tbl[["model", "n", "AUC (IC95%)", "brier",
           "cal_slope", "cal_intercept",
           "sens_youden", "spec_youden",
           "sens_paper", "spec_paper"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

tbl.to_csv(RESDIR / "comparacion_modelos_temporal.csv", index=False)
print(f"\nGuardado: {RESDIR/'comparacion_modelos_temporal.csv'}")

# 3. ROC ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for key, name in MODELS:
    y, p = preds[key]
    fpr, tpr, _ = roc_curve(y, p)
    auc = next(m["auc"] for m in metrics_list if m["model"] == key)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
            color=COLORS[key], lw=1.7)
ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
ax.set_xlabel("1 − especificidad")
ax.set_ylabel("Sensibilidad")
ax.set_title("Curvas ROC — validación temporal (test 2019+)")
ax.legend(loc="lower right", fontsize=9)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIGDIR / "roc_curves_temporal.png", dpi=150)
plt.close()
print(f"Guardado: {FIGDIR/'roc_curves_temporal.png'}")

# 4. Calibración -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Ideal")
for key, name in MODELS:
    y, p = preds[key]
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(prob_pred, prob_true, "o-", label=name,
            color=COLORS[key], markersize=4, lw=1.4)
ax.set_xlabel("Probabilidad predicha")
ax.set_ylabel("Proporción observada")
ax.set_title("Calibración — validación temporal (test 2019+)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIGDIR / "calibration_curves_temporal.png", dpi=150)
plt.close()
print(f"Guardado: {FIGDIR/'calibration_curves_temporal.png'}")

# 5. Decision curves ---------------------------------------------------
def net_benefit(y, p, threshold):
    pred = (p >= threshold).astype(int)
    n = len(y)
    if n == 0 or threshold >= 1:
        return np.nan
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp / n - fp / n * (threshold / (1 - threshold))

thresholds = np.linspace(0.005, 0.30, 60)
fig, ax = plt.subplots(figsize=(8, 5))
prev = preds[MODELS[0][0]][0].mean()
nb_treat_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
ax.plot(thresholds * 100, nb_treat_all, color="black", lw=1.0,
        label="Tratar a todos")
ax.axhline(0, color="black", lw=1.0, linestyle=":", label="Tratar a ninguno")

for key, name in MODELS:
    y, p = preds[key]
    nbs = [net_benefit(y, p, t) for t in thresholds]
    ax.plot(thresholds * 100, nbs, label=name, color=COLORS[key], lw=1.6)

ax.axvline(9.55, color="gray", linestyle="--", lw=0.8,
           label="Cutoff paper (9.55%)")
ax.set_xlabel("Umbral de probabilidad (%)")
ax.set_ylabel("Net benefit")
ax.set_title("Curva de decisión — validación temporal (test 2019+)")
ax.set_ylim(-0.02, 0.25)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGDIR / "decision_curves_temporal.png", dpi=150)
plt.close()
print(f"Guardado: {FIGDIR/'decision_curves_temporal.png'}")

print(f"\n=== Listo. Figuras en {FIGDIR} ===")

"""
14_consort_flowchart.py
-----------------------
CONSORT/TRIPOD+AI-style patient-flow diagram for the manuscript.
Output: analisis/figuras_temporal/figure1_consort.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT  = Path(__file__).resolve().parent.parent
FIGT  = ROOT / "analisis" / "figuras_temporal"
FIGT.mkdir(parents=True, exist_ok=True)

dat_full = pd.read_csv(ROOT / "analisis" / "datos_modelo.csv")
n_eligible      = len(dat_full)
n_with_outcome  = dat_full["cancer"].notna().sum()
no_outcome      = n_eligible - n_with_outcome
dat = dat_full.dropna(subset=["cancer"]).copy()
predictors = [
    "fhx_tc", "male", "age", "age2",
    "tsh_low", "tsh_high",
    "autoimmune", "solid", "susp_nodes", "hypoechoic",
    "irregular", "macrocalc", "microcalc", "taller_than_wide"
]
n_with_year = dat["year_surgery"].notna().sum()
no_year = len(dat) - n_with_year
dat = dat.dropna(subset=["year_surgery"])
n_after_year = len(dat)

n_complete = dat.dropna(subset=predictors).shape[0]
n_excl_miss = n_after_year - n_complete

dat_cc = dat.dropna(subset=predictors).copy()
n_train = (dat_cc["year_surgery"] < 2019).sum()
n_test  = (dat_cc["year_surgery"] >= 2019).sum()
ev_train = ((dat_cc["year_surgery"] < 2019) & (dat_cc["cancer"] == 1)).sum()
ev_test  = ((dat_cc["year_surgery"] >= 2019) & (dat_cc["cancer"] == 1)).sum()

print(f"Eligible records (CIRUGIA=1):                {n_eligible}")
print(f"  Excluded — missing outcome (TODOSCA):       {no_outcome}")
print(f"With histology outcome:                       {n_with_outcome}")
print(f"  Excluded — missing year of surgery:         {no_year}")
print(f"With year of surgery:                         {n_after_year}")
print(f"  Excluded — missing predictor(s):            {n_excl_miss}")
print(f"Analytic cohort (complete-case):              {n_complete}")
print(f"  Training set (surgery < 2019):              {n_train}  ({ev_train} cancers, {ev_train/n_train*100:.1f}%)")
print(f"  Test set     (surgery ≥ 2019):              {n_test}  ({ev_test} cancers, {ev_test/n_test*100:.1f}%)")


# -----------------------------------------------------------------
# Diagrama
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 13)
ax.axis("off")

box_kwargs = dict(facecolor="#EAF1FA", edgecolor="#1F3864", linewidth=1.5)
excl_kwargs = dict(facecolor="#FBE9D6", edgecolor="#B25E08", linewidth=1.2)

def box(x, y, w, h, text, fontsize=10, **kw):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
        **kw)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontfamily="DejaVu Sans")

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#1F3864",
                                lw=1.2, mutation_scale=14))

# Niveles
y0 = 12.2  # eligible
y1 = 10.0  # outcome
y2 = 7.8   # year
y3 = 5.6   # complete-case
y4 = 2.8   # split

# Cajas centrales
box(4, y0, 6, 1.0,
    f"Patients undergoing thyroidectomy\nat Hospital Universitario Puerto Real (2010–2023)\nn = {n_eligible}",
    **box_kwargs)
box(4, y1, 6, 1.0,
    f"With histologically confirmed outcome\n(thyroid cancer status)\nn = {n_with_outcome}",
    **box_kwargs)
box(4, y2, 6, 1.0,
    f"With recorded year of surgery\nn = {n_after_year}",
    **box_kwargs)
box(4, y3, 6, 1.1,
    f"Analytic cohort — complete data on all 13 predictors\n"
    f"n = {n_complete} ({n_complete/n_with_outcome*100:.1f}% of outcomes)",
    **box_kwargs)

# Cajas de exclusión a la derecha
box(8.4, (y0 + y1) / 2, 3.0, 0.9,
    f"Excluded — missing outcome\nn = {no_outcome}",
    fontsize=9, **excl_kwargs)
box(8.4, (y1 + y2) / 2, 3.0, 0.9,
    f"Excluded — missing year of surgery\nn = {no_year}",
    fontsize=9, **excl_kwargs)
box(8.4, (y2 + y3) / 2, 3.0, 1.1,
    f"Excluded — ≥1 predictor missing\nn = {n_excl_miss}\n(MICE sensitivity analysis\nin Section 3.4)",
    fontsize=9, **excl_kwargs)

# Cohortes finales
box(2.2, y4, 3.5, 1.6,
    f"Training cohort\nSurgery < 2019\nn = {n_train}\nThyroid cancer: {ev_train} ({ev_train/n_train*100:.1f}%)",
    **box_kwargs)
box(5.8, y4, 3.5, 1.6,
    f"Test cohort\nSurgery ≥ 2019\nn = {n_test}\nThyroid cancer: {ev_test} ({ev_test/n_test*100:.1f}%)",
    **box_kwargs)

# Flechas verticales
arrow(4, y0 - 0.5, 4, y1 + 0.5)
arrow(4, y1 - 0.5, 4, y2 + 0.5)
arrow(4, y2 - 0.5, 4, y3 + 0.55)

# Flechas a las cajas de exclusión
ax.plot([4, 6.9], [(y0 + y1)/2, (y0 + y1)/2], color="#1F3864", lw=1.0)
ax.plot([4, 6.9], [(y1 + y2)/2, (y1 + y2)/2], color="#1F3864", lw=1.0)
ax.plot([4, 6.9], [(y2 + y3)/2, (y2 + y3)/2], color="#1F3864", lw=1.0)

# Split temporal
arrow(4, y3 - 0.55, 2.2, y4 + 0.8)
arrow(4, y3 - 0.55, 5.8, y4 + 0.8)

# Etiqueta
box(4, y4 - 1.4, 6.5, 0.7,
    "Five models trained on training cohort, evaluated on test cohort\n"
    "(logistic regression, Random Forest, XGBoost, Bagged CART, MLP)",
    fontsize=9, facecolor="#E2EFDA", edgecolor="#385723", linewidth=1.2)

plt.tight_layout()
out_path = FIGT / "figure1_consort.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {out_path}")

"""
10_tabla1_comparacion_cohortes.py
---------------------------------
Paper-style Table 1: clinical characteristics compared between the
training (surgery before 2019) and test (2019 onwards) cohorts, with a
statistical test per variable (chi-square/Fisher for categorical;
Shapiro-Wilk then t-test or Mann-Whitney U for age).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT   = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "analisis" / "resultados_temporal"
RESDIR.mkdir(parents=True, exist_ok=True)

dat = pd.read_csv(ROOT / "analisis" / "datos_modelo_cc_temporal.csv")

train = dat[dat["cohort"] == "train"]
test  = dat[dat["cohort"] == "test"]
n_tr, n_te = len(train), len(test)
print(f"Train (pre-2019): n={n_tr}   |   Test (2019+): n={n_te}")

# ------------------------------------------------------------------
# Etiquetas amigables (mismo orden que la Tabla 1 del paper)
# ------------------------------------------------------------------
LABELS = [
    ("male",             "Sexo masculino, n (%)"),
    ("fhx_tc",           "Antecedente familiar de cáncer de tiroides, n (%)"),
    ("tsh_low",          "TSH < 0.369 µU/mL, n (%)"),
    ("tsh_high",         "TSH > 4.701 µU/mL, n (%)"),
    ("autoimmune",       "Tiroiditis autoinmune, n (%)"),
    ("solid",            "Nódulo sólido, n (%)"),
    ("susp_nodes",       "Adenopatías sospechosas, n (%)"),
    ("hypoechoic",       "Nódulo hipoecoico, n (%)"),
    ("irregular",        "Márgenes irregulares/microlobulados, n (%)"),
    ("macrocalc",        "Macrocalcificaciones, n (%)"),
    ("microcalc",        "Microcalcificaciones, n (%)"),
    ("taller_than_wide", "Más alto que ancho, n (%)"),
    ("cancer",           "Cáncer de tiroides (outcome), n (%)"),
]

def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"

rows = []

# ------------------------------------------------------------------
# Edad (continua)
# ------------------------------------------------------------------
age_tr, age_te = train["age"].values, test["age"].values
sw_tr = stats.shapiro(age_tr).pvalue
sw_te = stats.shapiro(age_te).pvalue
normal = (sw_tr > 0.05) and (sw_te > 0.05)
if normal:
    test_name = "t Student"
    pval = stats.ttest_ind(age_tr, age_te, equal_var=False).pvalue
    summary_tr = f"{age_tr.mean():.1f} ± {age_tr.std(ddof=1):.1f}"
    summary_te = f"{age_te.mean():.1f} ± {age_te.std(ddof=1):.1f}"
    label_age = "Edad, media ± DE (años)"
else:
    test_name = "Mann-Whitney U"
    pval = stats.mannwhitneyu(age_tr, age_te, alternative="two-sided").pvalue
    summary_tr = f"{np.median(age_tr):.0f} [{np.percentile(age_tr,25):.0f}–{np.percentile(age_tr,75):.0f}]"
    summary_te = f"{np.median(age_te):.0f} [{np.percentile(age_te,25):.0f}–{np.percentile(age_te,75):.0f}]"
    label_age = "Edad, mediana [P25–P75] (años)"

rows.append({
    "Variable":            label_age,
    f"Train (n={n_tr})":   summary_tr,
    f"Test (n={n_te})":    summary_te,
    "Test estadístico":    test_name,
    "p-valor":             fmt_p(pval),
})

# ------------------------------------------------------------------
# Categóricas
# ------------------------------------------------------------------
for var, label in LABELS:
    a = train[var].astype(int)
    b = test[var].astype(int)
    n1, n2 = a.sum(), b.sum()
    pc1, pc2 = n1 / n_tr * 100, n2 / n_te * 100
    table = np.array([[n1, n_tr - n1], [n2, n_te - n2]])
    expected = stats.chi2_contingency(table, correction=False).expected_freq
    if (expected < 5).any():
        test_name = "Fisher exacto"
        pval = stats.fisher_exact(table).pvalue
    else:
        test_name = "Chi-cuadrado"
        pval = stats.chi2_contingency(table, correction=False).pvalue
    rows.append({
        "Variable":            label,
        f"Train (n={n_tr})":   f"{n1} ({pc1:.1f}%)",
        f"Test (n={n_te})":    f"{n2} ({pc2:.1f}%)",
        "Test estadístico":    test_name,
        "p-valor":             fmt_p(pval),
    })

tabla = pd.DataFrame(rows)
print("\n=== Tabla 1 — Características clínicas por cohorte temporal ===\n")
print(tabla.to_string(index=False))

out_csv = RESDIR / "tabla1_comparacion_cohortes.csv"
tabla.to_csv(out_csv, index=False)
print(f"\nGuardado: {out_csv}")

"""
01_preparar_dataset.py
----------------------
Lee Calculadora 2024.sav, filtra CIRUGIA=1, selecciona las 13 variables del
modelo publicado + outcome (TODOSCA), construye dummies de TSH y guarda el
dataset analítico como CSV (para Python) y RDS-compatible vía feather (opcional).

Ejecutar desde la carpeta del proyecto:
    python analisis/01_preparar_dataset.py
"""

from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
SAV  = ROOT / "Calculadora 2024.sav"
OUT  = ROOT / "analisis" / "datos_modelo.csv"

# ---------------------------------------------------------------------------
# 1. Cargar
# ---------------------------------------------------------------------------
df, meta = pyreadstat.read_sav(str(SAV))
print(f"Cargado: {df.shape[0]} pacientes × {df.shape[1]} variables")

# ---------------------------------------------------------------------------
# 2. Filtrar cohorte: solo pacientes con cirugía (gold standard = AP definitiva)
# ---------------------------------------------------------------------------
df = df[df["CIRUGIA"] == 1].copy()
print(f"Tras filtrar CIRUGIA=1: {df.shape[0]} pacientes")

# ---------------------------------------------------------------------------
# 3. Variables del modelo publicado (Tabla 1)
#    Outcome: TODOSCA  (0 benigno / 1 cáncer)
# ---------------------------------------------------------------------------
out  = "TODOSCA"
predictores = [
    "AFCDT",              # Family history of TC
    "SEX",                # Gender (1 = male)
    "EDAD",               # Age
    "EDADCUADRADO",       # Squared age
    "TSHSUPRIMIDA",       # TSH 0–0.369
    "TSHALTA",            # TSH > 4.701
    "TIROIDITIS",         # Autoimmune thyroiditis
    "SOLIDO1",            # Solid nodule
    "GANGLIOSOSPECHOSO1", # Suspicious adenopathies
    "HIPOECOICO1",        # Hypoechoic
    "IRREGULAR1",         # Margins microlobed/irregular
    "MACROCA1",           # Macrocalcifications
    "MICROCA1",           # Microcalcifications
    "FORMA1",             # Taller than wide (1 = más alto que ancho)
]

# Renombrar a etiquetas en inglés (más cómodo para el código y la publicación)
rename = {
    "AFCDT":              "fhx_tc",
    "SEX":                "male",
    "EDAD":               "age",
    "EDADCUADRADO":       "age2",
    "TSHSUPRIMIDA":       "tsh_low",
    "TSHALTA":            "tsh_high",
    "TIROIDITIS":         "autoimmune",
    "SOLIDO1":            "solid",
    "GANGLIOSOSPECHOSO1": "susp_nodes",
    "HIPOECOICO1":        "hypoechoic",
    "IRREGULAR1":         "irregular",
    "MACROCA1":           "macrocalc",
    "MICROCA1":           "microcalc",
    "FORMA1":             "taller_than_wide",
    "TODOSCA":            "cancer",
    "AÑOCIRUGIA":         "year_surgery",
}

dat = df[predictores + [out, "AÑOCIRUGIA"]].rename(columns=rename)

# Corregir typo evidente: año 2921 → 2021
dat.loc[dat["year_surgery"] == 2921, "year_surgery"] = 2021

# Recalcular age2 desde age (en SPSS quedaron 14 NA aunque age estaba completo)
dat["age2"] = dat["age"] ** 2

# ---------------------------------------------------------------------------
# 4. Comprobar tipos y missings
# ---------------------------------------------------------------------------
print("\nMissings por variable:")
print(dat.isna().sum().sort_values(ascending=False))

# Casos completos (el artículo original presumiblemente usó complete-case)
n_pre = len(dat)
dat = dat.dropna(subset=["cancer"])  # outcome obligatorio
print(f"\nTras eliminar outcome NA: {len(dat)} (había {n_pre})")

# Tipos: outcome y categóricas como int para sklearn/ranger
for c in dat.columns:
    if c not in ("age", "age2", "year_surgery"):
        dat[c] = dat[c].astype("Int64")  # mantiene NA
dat["year_surgery"] = dat["year_surgery"].astype("Int64")

# ---------------------------------------------------------------------------
# 5. Resumen rápido
# ---------------------------------------------------------------------------
print("\nDistribución del outcome:")
print(dat["cancer"].value_counts(dropna=False))
print(f"Prevalencia de cáncer: {dat['cancer'].mean()*100:.1f}%")

print("\nMissings tras limpieza:")
miss = dat.isna().sum()
print(miss[miss > 0] if miss.sum() > 0 else "  (ninguno)")

# ---------------------------------------------------------------------------
# 6. Guardar dos versiones: completa (con NAs) y complete-case
# ---------------------------------------------------------------------------
dat.to_csv(OUT, index=False)
print(f"\nGuardado completo:      {OUT}  ({len(dat)} filas × {len(dat.columns)} columnas)")

OUT_CC = ROOT / "analisis" / "datos_modelo_cc.csv"
# Complete-case sobre las variables del modelo (no exigimos year_surgery aquí)
dat_cc = dat.dropna(subset=[c for c in dat.columns if c != "year_surgery"])
dat_cc.to_csv(OUT_CC, index=False)
print(f"Guardado complete-case: {OUT_CC}  ({len(dat_cc)} filas × {len(dat_cc.columns)} columnas)")
print(f"Pacientes perdidos por missings: {len(dat) - len(dat_cc)}")
print(f"Prevalencia de cáncer en complete-case: {dat_cc['cancer'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# 7. Cohorte para split temporal: requiere también year_surgery no-NA
# ---------------------------------------------------------------------------
OUT_TMP = ROOT / "analisis" / "datos_modelo_cc_temporal.csv"
dat_tmp = dat_cc.dropna(subset=["year_surgery"]).copy()
dat_tmp["cohort"] = (dat_tmp["year_surgery"] < 2019).map(
    {True: "train", False: "test"})
dat_tmp.to_csv(OUT_TMP, index=False)

print(f"\nGuardado split temporal: {OUT_TMP}  ({len(dat_tmp)} filas)")
print("\nCohorte temporal:")
print(dat_tmp.groupby(["cohort", "cancer"]).size().unstack(fill_value=0))
print("\nPrevalencia de cáncer por cohorte:")
print(dat_tmp.groupby("cohort")["cancer"].mean().mul(100).round(1))

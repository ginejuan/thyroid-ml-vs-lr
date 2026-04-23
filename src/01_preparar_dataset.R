# 01_preparar_dataset.R
# ---------------------------------------------------------------------
# Lee Calculadora 2024.sav, filtra CIRUGIA=1, selecciona las 13
# variables del modelo publicado + outcome (TODOSCA), recalcula age2 y
# guarda el dataset analítico como datos_modelo.rds (R) y .csv (Python).
#
# Equivalente exacto al script 01_preparar_dataset.py.
#
# Requiere: haven, dplyr
# ---------------------------------------------------------------------

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
})

ROOT <- here::here()  # o usar una ruta absoluta si no usas {here}
SAV  <- file.path(ROOT, "Calculadora 2024.sav")
OUT_RDS <- file.path(ROOT, "analisis", "datos_modelo.rds")
OUT_CSV <- file.path(ROOT, "analisis", "datos_modelo.csv")

# 1. Cargar -----------------------------------------------------------
df <- read_sav(SAV)
cat(sprintf("Cargado: %d pacientes x %d variables\n", nrow(df), ncol(df)))

# 2. Filtrar cohorte --------------------------------------------------
df <- df %>% filter(CIRUGIA == 1)
cat(sprintf("Tras filtrar CIRUGIA=1: %d pacientes\n", nrow(df)))

# 3. Variables del modelo + renombrado --------------------------------
dat <- df %>%
  transmute(
    fhx_tc           = AFCDT,
    male             = SEX,
    age              = EDAD,
    age2             = EDAD^2,                 # recalculado, SPSS tenía NAs
    tsh_low          = TSHSUPRIMIDA,
    tsh_high         = TSHALTA,
    autoimmune       = TIROIDITIS,
    solid            = SOLIDO1,
    susp_nodes       = GANGLIOSOSPECHOSO1,
    hypoechoic       = HIPOECOICO1,
    irregular        = IRREGULAR1,
    macrocalc        = MACROCA1,
    microcalc        = MICROCA1,
    taller_than_wide = FORMA1,
    cancer           = TODOSCA,
    year_surgery     = AÑOCIRUGIA
  ) %>%
  filter(!is.na(cancer)) %>%
  # Corregir typo evidente: año 2921 → 2021
  mutate(year_surgery = if_else(year_surgery == 2921, 2021, year_surgery))

# Convertir categóricas a factor (no a numéricas) para R/tidymodels
cat_vars <- c("fhx_tc","male","tsh_low","tsh_high","autoimmune",
              "solid","susp_nodes","hypoechoic","irregular",
              "macrocalc","microcalc","taller_than_wide","cancer")

dat <- dat %>%
  mutate(across(all_of(cat_vars), ~ factor(.x, levels = c(0, 1))))

# 4. Resumen ----------------------------------------------------------
cat("\nDistribución del outcome:\n")
print(table(dat$cancer, useNA = "always"))
cat(sprintf("Prevalencia de cáncer: %.1f%%\n",
            mean(dat$cancer == 1, na.rm = TRUE) * 100))

cat("\nMissings por variable:\n")
print(sort(colSums(is.na(dat)), decreasing = TRUE))

# 5. Guardar dos versiones: completa y complete-case ------------------
saveRDS(dat, OUT_RDS)
write.csv(dat, OUT_CSV, row.names = FALSE)

dat_cc <- na.omit(dat)
OUT_RDS_CC <- file.path(ROOT, "analisis", "datos_modelo_cc.rds")
OUT_CSV_CC <- file.path(ROOT, "analisis", "datos_modelo_cc.csv")
saveRDS(dat_cc, OUT_RDS_CC)
write.csv(dat_cc, OUT_CSV_CC, row.names = FALSE)

cat(sprintf("\nGuardado:\n  %s (n=%d)\n  %s (n=%d)\n",
            OUT_RDS, nrow(dat), OUT_RDS_CC, nrow(dat_cc)))
cat(sprintf("Pacientes perdidos por missings: %d\n", nrow(dat) - nrow(dat_cc)))
cat(sprintf("Prevalencia de cáncer en complete-case: %.1f%%\n",
            mean(dat_cc$cancer == 1) * 100))

# 6. Cohorte para split temporal: requiere también year_surgery no-NA -----
OUT_RDS_TMP <- file.path(ROOT, "analisis", "datos_modelo_cc_temporal.rds")
OUT_CSV_TMP <- file.path(ROOT, "analisis", "datos_modelo_cc_temporal.csv")

dat_tmp <- dat_cc %>%
  filter(!is.na(year_surgery)) %>%
  mutate(cohort = if_else(year_surgery < 2019, "train", "test"))

saveRDS(dat_tmp, OUT_RDS_TMP)
write.csv(dat_tmp, OUT_CSV_TMP, row.names = FALSE)

cat(sprintf("\nSplit temporal guardado: %s (n=%d)\n", OUT_RDS_TMP, nrow(dat_tmp)))
cat("\nTabla cohorte × cancer:\n")
print(table(dat_tmp$cohort, dat_tmp$cancer, useNA = "ifany"))
cat("\nPrevalencia de cáncer por cohorte (%):\n")
print(round(tapply(as.numeric(as.character(dat_tmp$cancer)),
                   dat_tmp$cohort, mean) * 100, 1))

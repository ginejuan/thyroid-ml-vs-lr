# 02_baseline_logistic.R
# ---------------------------------------------------------------------
# Baseline: regresión logística con los 13 predictores del modelo publicado.
# Genera coeficientes, predicciones por CV 10x5 y métricas (AUC, Brier,
# calibración, sens/esp en Youden y en el cutoff 9.55% del paper).
#
# Equivalente al script 02_baseline_logistic.py.
# Requiere: dplyr, pROC, rsample, yardstick, jsonlite
# ---------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
  library(rsample)
  library(yardstick)
  library(jsonlite)
})

ROOT   <- here::here()
DATCC  <- file.path(ROOT, "analisis", "datos_modelo_cc.rds")
RESDIR <- file.path(ROOT, "analisis", "resultados")
dir.create(RESDIR, showWarnings = FALSE, recursive = TRUE)

dat <- readRDS(DATCC)
cat(sprintf("n = %d, casos = %d (%.1f%%)\n",
            nrow(dat), sum(dat$cancer == 1),
            mean(dat$cancer == 1) * 100))

# Predictores en el orden de la Tabla 1
predictores <- c("fhx_tc","male","age","age2",
                 "tsh_low","tsh_high",
                 "autoimmune","solid","susp_nodes","hypoechoic",
                 "irregular","macrocalc","microcalc","taller_than_wide")

formula <- as.formula(paste("cancer ~", paste(predictores, collapse = " + ")))

# 1. Modelo en sample completo  --------------------------------------
mod <- glm(formula, data = dat, family = binomial)
coef_tbl <- broom::tidy(mod, exponentiate = FALSE) %>%
  mutate(OR = exp(estimate))
cat("\nCoeficientes (compárese con la Tabla 1 del paper):\n")
print(coef_tbl, digits = 3)
write.csv(coef_tbl, file.path(RESDIR, "baseline_logistic_coefs_R.csv"),
          row.names = FALSE)

# 2. CV 10x5 estratificada -------------------------------------------
SEED <- 20260422
set.seed(SEED)

n_folds   <- 10
n_repeats <- 5

proba_sum   <- numeric(nrow(dat))
proba_count <- numeric(nrow(dat))

for (rep in seq_len(n_repeats)) {
  set.seed(SEED + rep)
  splits <- vfold_cv(dat, v = n_folds, strata = cancer)
  for (i in seq_len(n_folds)) {
    sp  <- splits$splits[[i]]
    tr  <- analysis(sp)
    te  <- assessment(sp)
    idx <- as.integer(rownames(te))
    m   <- glm(formula, data = tr, family = binomial)
    p   <- predict(m, newdata = te, type = "response")
    proba_sum[idx]   <- proba_sum[idx]   + p
    proba_count[idx] <- proba_count[idx] + 1
  }
}
y_proba <- proba_sum / proba_count
y       <- as.integer(as.character(dat$cancer))

# 3. Métricas --------------------------------------------------------
auc   <- as.numeric(pROC::auc(y, y_proba))
brier <- mean((y - y_proba)^2)

# Calibración: glm de y sobre logit(p)
lp <- qlogis(pmin(pmax(y_proba, 1e-6), 1 - 1e-6))
cal <- glm(y ~ lp, family = binomial)
cal_intercept <- coef(cal)[1]
cal_slope     <- coef(cal)[2]

# Cutoffs
roc_obj <- pROC::roc(y, y_proba, direction = "<")
co_youden <- pROC::coords(roc_obj, "best", best.method = "youden",
                          ret = c("threshold","sensitivity","specificity"))
cutoff_paper <- 0.0955
sens_paper <- mean(y_proba[y == 1] >= cutoff_paper)
spec_paper <- mean(y_proba[y == 0] <  cutoff_paper)

metrics <- list(
  n              = length(y),
  cancer_n       = sum(y),
  cancer_prev    = mean(y),
  auc            = auc,
  brier          = brier,
  cal_slope      = unname(cal_slope),
  cal_intercept  = unname(cal_intercept),
  cutoff_youden  = unname(co_youden$threshold),
  sens_youden    = unname(co_youden$sensitivity),
  spec_youden    = unname(co_youden$specificity),
  cutoff_paper   = cutoff_paper,
  sens_paper     = sens_paper,
  spec_paper     = spec_paper,
  cv_seed        = SEED,
  cv_scheme      = "vfold_cv x 5 repeats"
)
cat("\nMétricas (CV 10x5):\n")
str(metrics, give.attr = FALSE)

write_json(metrics, file.path(RESDIR, "baseline_logistic_metrics_R.json"),
           auto_unbox = TRUE, pretty = TRUE)

write.csv(data.frame(y = y, p_logistic = y_proba),
          file.path(RESDIR, "baseline_logistic_predictions_R.csv"),
          row.names = FALSE)

cat(sprintf("\nResultados guardados en: %s\n", RESDIR))

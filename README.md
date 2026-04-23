# Machine learning vs logistic regression for malignancy risk in thyroid nodules

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19707930.svg)](https://doi.org/10.5281/zenodo.19707930)

Code, aggregate metrics and figures supporting the manuscript

> **Fernández Alba JJ, Carral San Laureano F, Jiménez Millán AI, González Macías C.**
> *Machine learning versus logistic regression for malignancy risk in thyroid nodules: a temporal validation study.*
> Journal of Clinical Epidemiology — in submission, 2026.

This repository accompanies a temporal-validation study comparing five classifiers for the binary outcome **"malignancy at definitive histology"** in thyroid nodules with indeterminate or suspicious cytology. The five models compared are: penalised logistic regression (the baseline, identical to the originally published model), random forest, XGBoost, bagged CART, and a multi-layer perceptron.

## Repository layout

```
.
├── README.md                          ← this file
├── LICENSE                            ← MIT — applies to /src
├── LICENSE-DATA                       ← CC-BY-4.0 — applies to /results, /figures
├── CITATION.cff                       ← machine-readable citation metadata
├── requirements.txt                   ← pinned Python dependencies
├── .gitignore                         ← excludes raw .sav and patient-level CSVs
│
├── src/                               ← analysis pipeline (R + Python)
│   ├── 01_preparar_dataset.R          ← variable engineering from the SPSS file
│   ├── 01_preparar_dataset.py         ← Python equivalent (used by all 0X scripts)
│   ├── 02_baseline_logistic.R         ← published logistic-regression baseline
│   ├── 02_baseline_logistic.py
│   ├── 03_random_forest.py
│   ├── 04_xgboost.py
│   ├── 05_bagged_cart.py
│   ├── 06_mlp.py
│   ├── 07_comparar_modelos.py         ← internal CV comparison
│   ├── 08_temporal_split.py           ← train ≤ 2019 / test 2020-2023
│   ├── 09_comparar_temporal.py        ← primary external comparison
│   ├── 10_tabla1_comparacion_cohortes.py
│   ├── 11_pruebas_estadisticas.py     ← DeLong, paired Brier, NRI/IDI
│   ├── 12_linealidad_interacciones.py ← RCS, two-way interactions
│   ├── 13_missing_y_imputacion.py     ← MICE sensitivity analysis
│   ├── 14_consort_flowchart.py
│   ├── 15_shap_importance.py          ← SHAP convergent-validity analysis
│   └── utils_cv.py                    ← shared CV / data-loading helpers
│
├── results/                           ← AGGREGATE-only outputs (no patient rows)
│   ├── *_metrics.json                 ← per-model summary metrics
│   ├── comparacion_modelos_temporal.csv
│   ├── tabla1_comparacion_cohortes.csv
│   ├── pruebas_estadisticas.csv       ← DeLong / paired-Brier results
│   ├── shap_importance_ranking.csv
│   ├── shap_concordancia.txt          ← Spearman ρ between SHAP and LR coefs
│   ├── linealidad_edad.csv            ← RCS test for non-linearity in age
│   ├── interacciones_dos_vias.csv
│   ├── missingness_summary.csv
│   ├── imputacion_multiple_resultado.csv
│   └── logistic_coefs_train.csv
│
├── figures/                           ← published figures
│   ├── figure1_consort.png
│   ├── roc_curves_temporal.png
│   ├── calibration_curves_temporal.png
│   ├── decision_curves_temporal.png
│   ├── figura_shap_summary.png
│   └── figura_shap_beeswarm.png
│
└── data/
    ├── README.md                      ← data access policy (LOPDGDD/GDPR)
    └── synthetic/
        └── generate_synthetic_data.py ← fake data generator for end-to-end demos
```

## Why no raw data?

The development cohort is a single-centre dataset from the Hospital Universitario Puerto Real (Cádiz, Spain), originally collected under research authorisation **PAI-TIROIDES-2018** (Biomedical Research Ethics Committee of Cádiz, April 2018). Even after stripping direct identifiers, each record retains quasi-identifiers — year of surgery, age and sex — that, in a single-centre cohort spanning 14 years, would carry a non-trivial re-identification risk. Public release of the row-level file is therefore not compatible with Spanish Organic Law 3/2018 (LOPDGDD) or EU Regulation 2016/679 (GDPR).

See [`data/README.md`](data/README.md) for the procedure to request a de-identified extract for replication or methodological re-analysis.

## Reproducing the pipeline with synthetic data

To verify that the code runs end-to-end without access to real patient data:

```bash
# 1. Set up a clean Python 3.11 environment
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate a fake but plausible dataset of the same shape
python data/synthetic/generate_synthetic_data.py \
    --n 2614 \
    --seed 42 \
    --out data/synthetic/datos_modelo_synthetic.csv

# 3. Point the scripts at the synthetic file and run the temporal pipeline
#    (each script reads ../data by default; see utils_cv.py for the path).
python src/08_temporal_split.py
python src/09_comparar_temporal.py
python src/11_pruebas_estadisticas.py
python src/15_shap_importance.py
```

Results computed on synthetic data are **not** substantively meaningful — they exist only to confirm that the pipeline produces non-degenerate plots, JSONs and CSVs.

## Reproducing the published numbers

With access to the real cohort (see data access policy):

```bash
# place the SPSS file as data/raw/Calculadora_2024.sav
Rscript src/01_preparar_dataset.R              # writes the analytic CSVs
python src/02_baseline_logistic.py             # baseline LR (internal CV)
python src/03_random_forest.py
python src/04_xgboost.py
python src/05_bagged_cart.py
python src/06_mlp.py
python src/07_comparar_modelos.py              # internal CV summary
python src/08_temporal_split.py                # train ≤2019 / test 2020-2023
python src/09_comparar_temporal.py             # primary external comparison
python src/10_tabla1_comparacion_cohortes.py
python src/11_pruebas_estadisticas.py          # DeLong, paired Brier, NRI/IDI
python src/12_linealidad_interacciones.py      # RCS / two-way interactions
python src/13_missing_y_imputacion.py          # MICE sensitivity
python src/14_consort_flowchart.py
python src/15_shap_importance.py
```

All random seeds are fixed in `src/utils_cv.py` (`SEED = 42`) and propagate to scikit-learn, XGBoost and the MLP.

## Software environment

- **Python:** 3.11 — full pinned list in [`requirements.txt`](requirements.txt).
- **R:** 4.3.x with packages `haven`, `dplyr`, `tidyr`, `glmnet`, `pROC`, `mice`, `rms`. The R scripts are kept for transparent reproduction of the originally published logistic-regression baseline; they share the input CSV with the Python pipeline.

## How to cite

If you use this code, the figures, or the aggregate metrics, please cite **both**:

1. The peer-reviewed manuscript (DOI to be added on acceptance).
2. This repository — see [`CITATION.cff`](CITATION.cff). Once a Zenodo release is created, a permanent DOI will be added here.

## Licences

- **Code (`/src`, `data/synthetic/`)** — MIT — see [`LICENSE`](LICENSE).
- **Aggregate results and figures (`/results`, `/figures`)** — Creative Commons Attribution 4.0 International — see [`LICENSE-DATA`](LICENSE-DATA).

## Contact

Juan Jesús Fernández Alba — `jjesus.fernandez@uca.es`
Department of Obstetrics and Gynaecology, Hospital Universitario Puerto Real
INiBICA · University of Cádiz · Spain

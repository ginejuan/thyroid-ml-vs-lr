# Data access policy


## Update (August 2026): minimal anonymised dataset now public

Following approval by the Biomedical Research Ethics Committee of Cádiz, a
**minimal anonymised dataset** sufficient to replicate all findings of the
temporal-validation study (complete-case n=861 and with-missing n=981, with
full codebook) has been deposited on Zenodo under a CC BY 4.0 licence:

> https://doi.org/10.5281/zenodo.22111356

De-identification measures: no direct identifiers or dates; surgery year
replaced by the analysis cohort label (train/test); nodule diameter replaced
by the size categories used in the subgroup analyses; prepared following
Hrynaszkiewicz et al. (BMJ 2010;340:c181). The restrictions described below
continue to apply to the **full row-level registry**, which retains
additional quasi-identifiers.

## Why the raw data are not in this repository

The development cohort (n = 2 614 thyroid nodules with cytology, ultrasound
features, demographic and biochemical data, plus histological outcome after
surgery) was assembled at the Hospital Universitario Puerto Real (Cádiz,
Spain) under research authorisation **PAI-TIROIDES-2018**, granted by the
Biomedical Research Ethics Committee of Cádiz in April 2018.

Even after stripping direct identifiers, each record retains
**quasi-identifiers** (year of surgery, age, sex, hospital of origin) that,
in combination, would create a non-trivial re-identification risk in a
single-centre cohort spanning 14 years. Public release of the row-level
file is therefore **not compatible** with:

- Spanish Organic Law 3/2018 (LOPDGDD), arts. 9 and 26.
- EU Regulation 2016/679 (GDPR), arts. 4(1), 4(5) and 9.
- The original IRB authorisation, which restricts use to the named
  research team for the stated objectives.

## How to request the data

Bona-fide researchers may request a de-identified extract for replication
or methodological re-analysis. Requests should be sent to the corresponding
author with:

1. A short scientific protocol (max. 1 page) describing the planned analysis.
2. The CV / institutional affiliation of the principal investigator.
3. A signed data-use agreement committing to:
   - non-redistribution of the row-level file,
   - no attempt at re-identification,
   - secure storage and deletion at the end of the project,
   - acknowledgement of the source cohort in any resulting publication.

Requests are reviewed by the corresponding author together with the
Biomedical Research Ethics Committee of Cádiz. Approved extracts are
shared via a secure transfer channel.

**Corresponding author:** Juan Jesús Fernández Alba —
`jjesus.fernandez@uca.es`

## What you *can* find in this repository

- `src/` — the full analysis pipeline (R + Python).
- `results/` — aggregate metrics (JSON, CSV) and the bibliographic / SHAP
  ranking tables. No row-level patient data.
- `figures/` — the published figures (CONSORT, ROC, calibration, decision
  curves, SHAP summary / beeswarm).
- `data/synthetic/` — a generator that produces a **fake but plausible**
  dataset of the same shape, so that reviewers can run the entire pipeline
  end-to-end without access to real patient data.

The synthetic data are generated from publicly reported marginal
distributions; they are **not** drawn from any real patient and must not
be interpreted as substantively meaningful — they exist only to verify
that the code runs.

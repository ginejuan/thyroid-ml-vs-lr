"""
generate_synthetic_data.py
==========================

Produce a *fake but plausible* dataset with the same column structure as the
real cohort, so that reviewers (and CI) can run the whole pipeline without
having access to patient-level data.

The synthetic data are NOT drawn from any real patient. They are generated
from independent marginal distributions chosen to roughly match published
prevalences and the descriptive table of the original article. **Do not**
interpret any number computed on this output as substantively meaningful.

Usage
-----
    python data/synthetic/generate_synthetic_data.py \\
        --n 2614 \\
        --seed 42 \\
        --out data/synthetic/datos_modelo_synthetic.csv

The output CSV has the same headers as `datos_modelo_cc_temporal.csv`:

    fhx_tc, male, age, age2, tsh_low, tsh_high, autoimmune, solid,
    susp_nodes, hypoechoic, irregular, macrocalc, microcalc,
    taller_than_wide, cancer, year_surgery, cohort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- approximate marginal prevalences (from the published cohort) -------
PREVS = {
    "fhx_tc":           0.05,   # family history of thyroid cancer
    "male":             0.20,
    "tsh_low":          0.06,
    "tsh_high":         0.10,
    "autoimmune":       0.18,
    "solid":            0.55,
    "susp_nodes":       0.04,
    "hypoechoic":       0.45,
    "irregular":        0.15,
    "macrocalc":        0.10,
    "microcalc":        0.18,
    "taller_than_wide": 0.12,
}

# logistic model used to assign a plausible cancer label;
# coefficients are ROUGHLY in line with published ORs but are NOT the real
# fitted coefficients — they are here only to make the synthetic outcome
# correlate with the predictors so the pipeline produces non-degenerate plots.
COEFS = {
    "intercept":        -3.2,
    "fhx_tc":            0.50,
    "male":              0.40,
    "age_z":             0.10,    # standardised age
    "tsh_low":          -0.30,
    "tsh_high":          0.45,
    "autoimmune":        0.25,
    "solid":             0.60,
    "susp_nodes":        1.20,
    "hypoechoic":        0.50,
    "irregular":         0.90,
    "macrocalc":         0.20,
    "microcalc":         0.80,
    "taller_than_wide":  1.00,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # binary predictors
    df = pd.DataFrame({
        col: rng.binomial(1, p, size=n).astype(int)
        for col, p in PREVS.items()
    })

    # age — truncated normal around 52, sd 14, clipped to [18, 90]
    age = rng.normal(loc=52.0, scale=14.0, size=n)
    age = np.clip(age, 18, 90).round().astype(int)
    df["age"] = age
    df["age2"] = age ** 2

    # year of surgery — uniform 2010..2023 (matches the temporal split logic)
    df["year_surgery"] = rng.integers(low=2010, high=2024, size=n)

    # cohort = "train" if year < 2020 else "test", same convention as the paper
    df["cohort"] = np.where(df["year_surgery"] < 2020, "train", "test")

    # generate cancer label from a logistic model
    age_z = (df["age"] - df["age"].mean()) / df["age"].std()
    logit = (
        COEFS["intercept"]
        + COEFS["fhx_tc"]            * df["fhx_tc"]
        + COEFS["male"]              * df["male"]
        + COEFS["age_z"]             * age_z
        + COEFS["tsh_low"]           * df["tsh_low"]
        + COEFS["tsh_high"]          * df["tsh_high"]
        + COEFS["autoimmune"]        * df["autoimmune"]
        + COEFS["solid"]             * df["solid"]
        + COEFS["susp_nodes"]        * df["susp_nodes"]
        + COEFS["hypoechoic"]        * df["hypoechoic"]
        + COEFS["irregular"]         * df["irregular"]
        + COEFS["macrocalc"]         * df["macrocalc"]
        + COEFS["microcalc"]         * df["microcalc"]
        + COEFS["taller_than_wide"]  * df["taller_than_wide"]
    )
    p = _sigmoid(logit.values)
    df["cancer"] = rng.binomial(1, p).astype(int)

    # final column order — must match datos_modelo_cc_temporal.csv
    cols = [
        "fhx_tc", "male", "age", "age2", "tsh_low", "tsh_high",
        "autoimmune", "solid", "susp_nodes", "hypoechoic", "irregular",
        "macrocalc", "microcalc", "taller_than_wide",
        "cancer", "year_surgery", "cohort",
    ]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",    type=int, default=2614,
                        help="number of synthetic patients to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--out",  type=Path,
                        default=Path("data/synthetic/datos_modelo_synthetic.csv"),
                        help="output CSV path")
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    prev = df["cancer"].mean()
    print(f"Wrote {len(df)} synthetic rows to {args.out}")
    print(f"Synthetic cancer prevalence: {prev:.3f} "
          f"(real cohort ≈ 0.10–0.13; for shape-checking only).")


if __name__ == "__main__":
    main()

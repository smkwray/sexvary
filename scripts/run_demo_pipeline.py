#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import numpy as np
import pandas as pd

from sexvary.meta import dersimonian_laird_meta
from sexvary.metrics import log_variance_ratio_from_groups, tail_rate_ratio_from_groups
from sexvary.reporting import forest_plot_from_effects
from sexvary.utils import ensure_dir, project_root


def main() -> None:
    root = project_root(__file__)
    out_dir = ensure_dir(root / "results" / "demo")

    rng = np.random.default_rng(20260314)
    synthetic_cells = []

    specs = [
        ("nlsy79_main", "general_intelligence_g", 1.12, 0.0),
        ("piaac_cycle2", "numeracy", 1.08, 0.1),
        ("ecls_k_2011", "math_achievement", 1.05, -0.05),
        ("hsls_2009", "algebraic_reasoning", 1.09, 0.15),
    ]

    for dataset_id, trait_id, male_sd, mean_shift in specs:
        female = rng.normal(loc=0.0, scale=1.0, size=6000)
        male = rng.normal(loc=mean_shift, scale=male_sd, size=6000)
        w_f = rng.uniform(0.5, 1.5, size=female.size)
        w_m = rng.uniform(0.5, 1.5, size=male.size)

        vr, md = log_variance_ratio_from_groups(
            male,
            female,
            numerator_weights=w_m,
            denominator_weights=w_f,
            numerator_label="male",
            denominator_label="female",
        )
        tail = tail_rate_ratio_from_groups(
            male,
            female,
            numerator_weights=w_m,
            denominator_weights=w_f,
            quantile=0.95,
            tail="upper",
        )
        synthetic_cells.append(
            {
                "dataset_id": dataset_id,
                "trait_id": trait_id,
                "log_variance_ratio": vr.log_variance_ratio,
                "variance_ratio": vr.variance_ratio,
                "mean_difference": md.mean_difference,
                "smd": md.standardized_mean_difference,
                "upper95_tail_ratio": tail.rate_ratio,
                "upper95_representation_ratio": tail.representation_ratio,
                "se_demo": abs(vr.log_variance_ratio) / 8.0 + 0.02,
            }
        )

    df = pd.DataFrame(synthetic_cells)
    df.to_csv(out_dir / "demo_trait_estimates.csv", index=False)

    meta = dersimonian_laird_meta(df["log_variance_ratio"], np.square(df["se_demo"]))
    pd.DataFrame(
        [
            {
                "model": meta.model,
                "k": meta.k,
                "estimate": meta.estimate,
                "estimate_backtransformed": meta.estimate_backtransformed,
                "se": meta.standard_error,
                "ci_low": meta.ci_low,
                "ci_high": meta.ci_high,
                "tau2": meta.tau2,
                "i2": meta.i2,
            }
        ]
    ).to_csv(out_dir / "demo_meta_analysis.csv", index=False)

    forest_plot_from_effects(
        df.assign(label=df["dataset_id"] + " :: " + df["trait_id"]),
        label_col="label",
        effect_col="log_variance_ratio",
        se_col="se_demo",
        output_path=out_dir / "demo_forest_plot.png",
        title="Synthetic demo: log variance ratios",
    )

    print(f"Wrote demo outputs to {out_dir}")


if __name__ == "__main__":
    main()

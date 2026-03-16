import pandas as pd

from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence


def test_annotate_estimate_evidence_marks_method_limited_simple_design() -> None:
    df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_main"],
            "trait_id": ["afqt"],
            "log_variance_ratio": [0.2],
            "se_log_variance_ratio": [0.05],
            "inference_method": ["analytic_effective_n_simple_design"],
            "qa_flags": [pd.NA],
        }
    )
    out = annotate_estimate_evidence(df, registry=build_registry())
    row = out.iloc[0]
    assert row["evidence_status"] == "method_limited"
    assert not row["headline_eligible"]
    assert row["method_limited"]
    assert row["comparability_tier"] == "confirmatory_method_limited"


def test_annotate_estimate_evidence_prioritizes_provisional_over_method_limited() -> None:
    df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_child_ya"],
            "trait_id": ["ppvt"],
            "log_variance_ratio": [0.1],
            "se_log_variance_ratio": [0.08],
            "inference_method": ["analytic_effective_n_simple_design"],
            "qa_flags": ["nonprimary_weight_fallback;mixed_weight_sources"],
        }
    )
    out = annotate_estimate_evidence(df, registry=build_registry())
    row = out.iloc[0]
    assert row["evidence_status"] == "provisional"
    assert row["provisional"]
    assert not row["headline_eligible"]
    assert row["suppression_reason"] == "nonprimary_weight_fallback;mixed_weight_sources"


def test_annotate_estimate_evidence_marks_qa_only_rows() -> None:
    df = pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023"],
            "trait_id": ["adult_cognition_screen"],
            "log_variance_ratio": [pd.NA],
            "se_log_variance_ratio": [pd.NA],
            "inference_method": ["unavailable"],
            "qa_flags": ["low_n_variance;tail_metrics_suppressed"],
        }
    )
    out = annotate_estimate_evidence(df, registry=build_registry())
    row = out.iloc[0]
    assert row["evidence_status"] == "qa_only"
    assert row["qa_only"]
    assert row["suppression_reason"] == "low_n_variance"

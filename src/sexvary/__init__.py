"""sexvary: seeded utilities for descriptive analysis of sex differences in variability."""

from .config import (
    DatasetSpec,
    TraitSpec,
    build_registry,
    load_analysis_config,
    load_datasets,
    load_local_paths,
    normalize_local_dataset_path,
    resolve_local_dataset_path,
    load_traits,
)
from .metrics import (
    MeanDifferenceResult,
    TailRatioResult,
    VarianceRatioResult,
    effective_sample_size,
    log_variance_ratio_from_groups,
    tail_rate_ratio_from_groups,
    weighted_mean,
    weighted_quantile,
    weighted_var,
)
from .meta import MetaAnalysisResult, dersimonian_laird_meta, fixed_effect_meta
from .piaac import detect_piaac_replicate_cols, estimate_piaac_cells, infer_piaac_replicate_spec
from .pisa import detect_pisa_replicate_cols, estimate_pisa_cells, infer_pisa_replicate_spec
from .timss import estimate_timss_cells, infer_timss_replicate_spec
from .pv_replicate import ReplicateDesignSpec, estimate_pv_replicate_cells, infer_replicate_design
from .estimation import (
    EstimationConfig,
    derive_age_band,
    estimate_dataset_cells,
    estimate_sex_difference_cell,
    estimation_config_from_analysis,
    prepare_analysis_frame,
)
from .survey import (
    jackknife_zone_replicate_estimates,
    combine_plausible_values,
    combine_plausible_values_and_replicates,
    replicate_variance,
    stratified_cluster_bootstrap_variance,
)

__all__ = [
    "DatasetSpec",
    "TraitSpec",
    "build_registry",
    "load_analysis_config",
    "load_datasets",
    "load_local_paths",
    "normalize_local_dataset_path",
    "resolve_local_dataset_path",
    "load_traits",
    "MeanDifferenceResult",
    "TailRatioResult",
    "VarianceRatioResult",
    "effective_sample_size",
    "log_variance_ratio_from_groups",
    "tail_rate_ratio_from_groups",
    "weighted_mean",
    "weighted_quantile",
    "weighted_var",
    "MetaAnalysisResult",
    "dersimonian_laird_meta",
    "fixed_effect_meta",
    "detect_piaac_replicate_cols",
    "estimate_piaac_cells",
    "infer_piaac_replicate_spec",
    "detect_pisa_replicate_cols",
    "estimate_pisa_cells",
    "infer_pisa_replicate_spec",
    "estimate_timss_cells",
    "infer_timss_replicate_spec",
    "ReplicateDesignSpec",
    "estimate_pv_replicate_cells",
    "infer_replicate_design",
    "EstimationConfig",
    "derive_age_band",
    "estimate_dataset_cells",
    "estimate_sex_difference_cell",
    "estimation_config_from_analysis",
    "prepare_analysis_frame",
    "combine_plausible_values",
    "combine_plausible_values_and_replicates",
    "replicate_variance",
    "stratified_cluster_bootstrap_variance",
    "jackknife_zone_replicate_estimates",
]

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_dataset import (
    SUPPORTING_EVIDENCE_DATASET_IDS,
    build_age_profile_summary,
    build_closest_to_equal_cells,
    build_dataset_distribution_summary,
    build_dataset_inventory,
    build_largest_n_cells,
    build_priority_summary,
    build_strongest_female_greater_cells,
    build_strongest_male_greater_cells,
    build_supporting_evidence_summary,
    build_supporting_evidence_top_cells,
    build_top_cells,
    build_trait_family_summary,
    build_widest_ci_cells,
    ensure_display_ready_columns,
)
from .reporting import markdown_table


PUBLIC_SITE_PAGES = ("index.html", "results.html", "datasets.html")


def _clean_value(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    return [
        {column: _clean_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _format_int(value: object) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{int(value):,}"


def _format_pct(value: object, *, digits: int = 0) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{100.0 * float(value):.{digits}f}%"


def _format_vr(value: object, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}x"


def _format_decimal(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def _humanize_token(value: object) -> str:
    if value is None:
        return "—"
    text = str(value)
    if not text:
        return "—"
    if "_" not in text:
        return text
    text = text.replace("_", " ")
    text = text.replace(" ya", " YA")
    text = text.replace(" psid", " PSID")
    return text.title().replace("K ", "K ").replace("Piaac", "PIAAC").replace("Nlsy", "NLSY")


def _page_title(title: str) -> str:
    return f"{title} — Sex Differences in Variability" if title else "Sex Differences in Variability"


def _badge_class(claim_status_display: object) -> str:
    mapping = {
        "Headline claim": "badge--headline",
        "Supporting evidence": "badge--supporting",
        "Provisional": "badge--provisional",
        "Method-limited": "badge--method",
        "QA only": "badge--qa",
    }
    return mapping.get(str(claim_status_display), "badge--qa")


def _direction_badge_class(direction: object) -> str:
    mapping = {
        "male_greater": "badge--headline",
        "female_greater": "badge--supporting",
        "near_equal": "badge--qa",
    }
    return mapping.get(str(direction), "badge--qa")


def _direction_text(direction: object) -> str:
    mapping = {
        "male_greater": "M > F",
        "female_greater": "F > M",
        "near_equal": "Near equal",
        "unavailable": "Unavailable",
    }
    return mapping.get(str(direction), str(direction))


def _cell_label(row: dict[str, object]) -> str:
    return " | ".join(
        [
            str(row.get("dataset_label", "—")),
            str(row.get("trait_label", "—")),
            str(row.get("age_band", "—")),
            _humanize_token(row.get("cycle_or_wave")),
        ]
    )


def _dataset_claim_status(sub: pd.DataFrame) -> str:
    confirmatory_headline = sub[(sub["headline_eligible"]) & (sub["trait_priority"] == "confirmatory")]
    if not confirmatory_headline.empty:
        return "Headline claim"
    supporting = sub[sub["claim_status_display"] == "Supporting evidence"]
    if not supporting.empty:
        return "Supporting evidence"
    provisional = sub[sub["provisional"]]
    if not provisional.empty:
        return "Provisional"
    method_limited = sub[sub["method_limited"]]
    if not method_limited.empty:
        return "Method-limited"
    return "QA only"


def _augment_inventory(inventory: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in inventory.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        sub = comparison_df[comparison_df["dataset_id"] == dataset_id].copy()
        row["dataset_claim_status"] = _dataset_claim_status(sub)
        row["median_variance_ratio"] = float(np.exp(row["median_log_variance_ratio"])) if pd.notna(row["median_log_variance_ratio"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset_label", kind="stable").reset_index(drop=True)


def _chart_forest_records(df: pd.DataFrame, *, limit: int) -> list[dict[str, object]]:
    rows = _records(df.head(limit))
    return [
        {
            "label": _cell_label(row),
            "vr": row.get("variance_ratio"),
            "ciLow": row.get("vr_ci_low"),
            "ciHigh": row.get("vr_ci_high"),
        }
        for row in rows
    ]


def _chart_hbar_records(df: pd.DataFrame, *, limit: int = 18) -> list[dict[str, object]]:
    if df.empty:
        return []
    work = df.copy()
    work = work[np.isfinite(work["median_log_variance_ratio"])].copy()
    if work.empty:
        return []
    work["median_variance_ratio"] = np.exp(work["median_log_variance_ratio"])
    work["label"] = work["dataset_label"] + " | " + work["trait_family"].astype(str).map(_humanize_token)
    work = work.sort_values("median_log_variance_ratio", kind="stable").tail(limit)
    return _records(work[["label", "median_variance_ratio"]].rename(columns={"median_variance_ratio": "vr"}))


def _chart_age_records(age_profile: pd.DataFrame) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for dataset_label, sub in age_profile.groupby("dataset_label", sort=True):
        groups.append(
            {
                "name": dataset_label,
                "points": [
                    {
                        "label": str(row["age_band"]),
                        "logvr": float(row["median_log_variance_ratio"]),
                    }
                    for _, row in sub.iterrows()
                ],
            }
        )
    return groups


def _chart_robustness_records(robustness: pd.DataFrame) -> list[dict[str, object]]:
    if robustness.empty:
        return []
    work = robustness.copy()
    work["label"] = work["dataset_label"] + " | " + work["variant"].astype(str)
    return _records(
        work[["label", "median_abs_delta", "sign_change_rate"]].rename(
            columns={"median_abs_delta": "delta", "sign_change_rate": "sign"}
        )
    )


def build_site_bundle(
    comparison_df: pd.DataFrame,
    *,
    robustness_summary: pd.DataFrame | None = None,
    supporting_robustness: pd.DataFrame | None = None,
) -> dict[str, object]:
    comparison_df = ensure_display_ready_columns(comparison_df)
    robustness_summary = robustness_summary if robustness_summary is not None else pd.DataFrame()
    supporting_robustness = supporting_robustness if supporting_robustness is not None else pd.DataFrame()

    dataset_inventory = _augment_inventory(build_dataset_inventory(comparison_df), comparison_df)
    trait_family_summary = build_trait_family_summary(comparison_df)
    priority_summary = build_priority_summary(comparison_df)
    supporting_summary = build_supporting_evidence_summary(comparison_df)
    age_profile_summary = build_age_profile_summary(comparison_df)
    dataset_distribution = build_dataset_distribution_summary(comparison_df)

    headline_confirmatory_all = build_top_cells(
        comparison_df[comparison_df["headline_eligible"]],
        limit=int(len(comparison_df)),
        priorities=("confirmatory",),
    )
    strongest_male_greater = build_strongest_male_greater_cells(comparison_df, limit=12)
    strongest_female_greater = build_strongest_female_greater_cells(comparison_df, limit=12)
    closest_to_equal = build_closest_to_equal_cells(comparison_df, limit=12)
    largest_n = build_largest_n_cells(comparison_df, limit=12)
    widest_ci = build_widest_ci_cells(comparison_df, limit=12)
    supporting_top = build_supporting_evidence_top_cells(comparison_df, limit=12)

    inferential_cells = comparison_df[comparison_df["ci_available"]].copy()
    public_cells = comparison_df[comparison_df["effect_available"]].copy()
    public_cells["ci_width"] = public_cells["vr_ci_high"] - public_cells["vr_ci_low"]
    supporting_inferential = comparison_df[
        comparison_df["dataset_id"].isin(SUPPORTING_EVIDENCE_DATASET_IDS) & comparison_df["ci_available"]
    ].copy()

    headline_positive_candidates = headline_confirmatory_all[headline_confirmatory_all["direction"] == "male_greater"].copy()
    headline_negative_candidates = headline_confirmatory_all[headline_confirmatory_all["direction"] == "female_greater"].copy()

    strongest_positive = (
        headline_positive_candidates.sort_values("log_variance_ratio", ascending=False, kind="stable").iloc[0].to_dict()
        if not headline_positive_candidates.empty
        else (headline_confirmatory_all.iloc[0].to_dict() if not headline_confirmatory_all.empty else None)
    )
    strongest_negative = (
        headline_negative_candidates.sort_values("log_variance_ratio", ascending=True, kind="stable").iloc[0].to_dict()
        if not headline_negative_candidates.empty
        else (headline_confirmatory_all.sort_values("log_variance_ratio", kind="stable").iloc[0].to_dict() if not headline_confirmatory_all.empty else None)
    )

    summary = {
        "live_dataset_count": int(dataset_inventory["dataset_id"].nunique()),
        "analysis_cell_count": int(len(comparison_df)),
        "inferential_cell_count": int(comparison_df["ci_available"].sum()),
        "headline_confirmatory_cell_count": int(len(headline_confirmatory_all)),
        "headline_dataset_count": int(headline_confirmatory_all["dataset_id"].nunique()) if not headline_confirmatory_all.empty else 0,
        "headline_positive_share": float((headline_confirmatory_all["direction"] == "male_greater").mean()) if not headline_confirmatory_all.empty else 0.0,
        "headline_median_variance_ratio": float(headline_confirmatory_all["variance_ratio"].median()) if not headline_confirmatory_all.empty else None,
        "headline_mean_variance_ratio": float(headline_confirmatory_all["variance_ratio"].mean()) if not headline_confirmatory_all.empty else None,
        "headline_min_variance_ratio": float(headline_confirmatory_all["variance_ratio"].min()) if not headline_confirmatory_all.empty else None,
        "headline_max_variance_ratio": float(headline_confirmatory_all["variance_ratio"].max()) if not headline_confirmatory_all.empty else None,
        "supporting_inferential_cell_count": int(len(supporting_inferential)),
        "provisional_cell_count": int(comparison_df["provisional"].sum()),
        "method_limited_cell_count": int(comparison_df["method_limited"].sum()),
        "qa_only_cell_count": int(comparison_df["qa_only"].sum()),
    }

    page_metrics = {
        "home": {
            "headline_confirmatory_cell_count": summary["headline_confirmatory_cell_count"],
            "headline_positive_share": summary["headline_positive_share"],
            "live_dataset_count": summary["live_dataset_count"],
            "supporting_inferential_cell_count": summary["supporting_inferential_cell_count"],
        },
        "results": {
            "headline_confirmatory_cell_count": summary["headline_confirmatory_cell_count"],
            "supporting_inferential_cell_count": summary["supporting_inferential_cell_count"],
            "inferential_cell_count": summary["inferential_cell_count"],
        },
        "datasets": {
            "inventory_row_count": int(len(dataset_inventory)),
            "headline_dataset_count": summary["headline_dataset_count"],
            "supporting_dataset_count": int(
                dataset_inventory["dataset_claim_status"].isin(["Supporting evidence", "Provisional", "Method-limited"]).sum()
            ),
        },
        "readme": {
            "headline_confirmatory_cell_count": summary["headline_confirmatory_cell_count"],
            "headline_positive_share": summary["headline_positive_share"],
            "live_dataset_count": summary["live_dataset_count"],
            "supporting_inferential_cell_count": summary["supporting_inferential_cell_count"],
        },
    }

    bundle = {
        "summary": summary,
        "page_metrics": page_metrics,
        "key_cells": {
            "headline_positive": _clean_value(strongest_positive) if strongest_positive is None else {k: _clean_value(v) for k, v in strongest_positive.items()},
            "headline_counterexample": _clean_value(strongest_negative) if strongest_negative is None else {k: _clean_value(v) for k, v in strongest_negative.items()},
        },
        "tables": {
            "dataset_inventory": _records(dataset_inventory),
            "trait_family_summary": _records(trait_family_summary),
            "priority_summary": _records(priority_summary),
            "supporting_summary": _records(supporting_summary),
            "dataset_distribution": _records(dataset_distribution),
            "headline_confirmatory_all": _records(headline_confirmatory_all),
            "headline_confirmatory_top": _records(headline_confirmatory_all.head(18)),
            "strongest_male_greater": _records(strongest_male_greater),
            "strongest_female_greater": _records(strongest_female_greater),
            "closest_to_equal": _records(closest_to_equal),
            "largest_n_cells": _records(largest_n),
            "widest_ci_cells": _records(widest_ci),
            "supporting_top_cells": _records(supporting_top),
            "inferential_cells": _records(
                inferential_cells.sort_values(
                    ["claim_status_display", "dataset_label", "trait_label", "cycle_or_wave", "age_band"],
                    kind="stable",
                )[
                    [
                        "dataset_label",
                        "cycle_or_wave",
                        "age_band",
                        "trait_label",
                        "claim_status_display",
                        "direction",
                        "variance_ratio",
                        "vr_ci_low",
                        "vr_ci_high",
                        "n_total",
                        "male_n",
                        "female_n",
                        "display_explanation",
                    ]
                ]
            ),
            "all_public_cells": _records(
                public_cells.sort_values(
                    ["dataset_label", "trait_label", "cycle_or_wave", "age_band"],
                    kind="stable",
                )
            ),
            "explorer_cells": _records(
                public_cells.sort_values(
                    ["dataset_label", "trait_label", "cycle_or_wave", "age_band"],
                    kind="stable",
                )[
                    [
                        "dataset_label",
                        "cycle_or_wave",
                        "age_band",
                        "trait_label",
                        "claim_status_display",
                        "direction",
                        "variance_ratio",
                        "vr_ci_low",
                        "vr_ci_high",
                        "ci_width",
                        "n_total",
                        "abs_log_vr",
                        "distance_from_equal",
                        "display_explanation",
                    ]
                ]
            ),
            "robustness_summary": _records(robustness_summary),
            "supporting_robustness": _records(supporting_robustness),
            "age_profile_summary": _records(age_profile_summary),
        },
        "charts": {
            "headline_forest": _chart_forest_records(headline_confirmatory_all, limit=18),
            "dataset_family_hbar": _chart_hbar_records(trait_family_summary),
            "age_profile": _chart_age_records(age_profile_summary),
            "robustness": _chart_robustness_records(robustness_summary),
        },
    }
    validate_site_bundle(bundle)
    return bundle


def validate_site_bundle(bundle: dict[str, object]) -> None:
    summary = dict(bundle.get("summary", {}))
    page_metrics = dict(bundle.get("page_metrics", {}))
    tables = dict(bundle.get("tables", {}))

    dataset_inventory = list(tables.get("dataset_inventory", []))
    headline_confirmatory_all = list(tables.get("headline_confirmatory_all", []))
    supporting_summary = list(tables.get("supporting_summary", []))
    inferential_cells = list(tables.get("inferential_cells", []))

    expected_supporting = sum(int(row.get("rows_with_ci") or 0) for row in supporting_summary)
    checks = [
        (
            int(summary.get("live_dataset_count", -1)),
            len(dataset_inventory),
            "summary.live_dataset_count",
        ),
        (
            int(summary.get("headline_confirmatory_cell_count", -1)),
            len(headline_confirmatory_all),
            "summary.headline_confirmatory_cell_count",
        ),
        (
            int(summary.get("supporting_inferential_cell_count", -1)),
            expected_supporting,
            "summary.supporting_inferential_cell_count",
        ),
        (
            int(summary.get("inferential_cell_count", -1)),
            len(inferential_cells),
            "summary.inferential_cell_count",
        ),
        (
            int(page_metrics.get("home", {}).get("headline_confirmatory_cell_count", -1)),
            int(summary.get("headline_confirmatory_cell_count", -2)),
            "page_metrics.home.headline_confirmatory_cell_count",
        ),
        (
            int(page_metrics.get("results", {}).get("headline_confirmatory_cell_count", -1)),
            int(summary.get("headline_confirmatory_cell_count", -2)),
            "page_metrics.results.headline_confirmatory_cell_count",
        ),
        (
            int(page_metrics.get("readme", {}).get("headline_confirmatory_cell_count", -1)),
            int(summary.get("headline_confirmatory_cell_count", -2)),
            "page_metrics.readme.headline_confirmatory_cell_count",
        ),
        (
            int(page_metrics.get("datasets", {}).get("inventory_row_count", -1)),
            len(dataset_inventory),
            "page_metrics.datasets.inventory_row_count",
        ),
    ]
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"Site bundle consistency failure for {label}: expected {expected}, found {actual}")


def write_site_bundle_json(bundle: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def _nav(active: str) -> str:
    pages = [
        ("index.html", "Home"),
        ("results.html", "Results"),
        ("datasets.html", "Datasets"),
        ("methods.html", "Methods"),
        ("limits.html", "Limits"),
        ("explanations.html", "Explanations"),
    ]
    link_rows: list[str] = []
    for href, label in pages:
        active_class = ' class="active"' if href == active else ""
        link_rows.append(f'      <li><a href="{href}"{active_class}>{label}</a></li>')
    links = "\n".join(link_rows)
    return f"""<nav class="nav">
  <div class="nav__inner">
    <a href="index.html" class="nav__brand">sexvary</a>
    <button class="nav__toggle" aria-label="Toggle navigation" aria-expanded="false">&#9776;</button>
    <ul class="nav__links">
{links}
      <li><button class="theme-toggle" aria-label="Toggle dark mode"></button></li>
    </ul>
  </div>
</nav>"""


def _page_shell(
    *,
    title: str,
    active: str,
    description: str,
    body: str,
    charts_script: str = "",
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(_page_title(title))}</title>
  <meta name="description" content="{html.escape(description)}">
  <link rel="stylesheet" href="assets/css/main.css">
  <script src="assets/js/main.js"></script>
</head>
<body>
{_nav(active)}
{body}
<footer class="footer">
  <div class="container">
    This project is descriptive, not causal. It does not establish biological or essentialist conclusions.
  </div>
</footer>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="assets/js/charts.js"></script>
{charts_script}
</body>
</html>
"""


def _html_badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{html.escape(text)}</span>'


def _html_table(headers: list[str], rows: list[list[str]], *, caption: str | None = None) -> str:
    caption_html = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    head_html = "".join(f"<th>{header}</th>" for header in headers)
    body_html = "\n".join(
        "<tr>" + "".join(cell for cell in row) + "</tr>"
        for row in rows
    )
    return f"""<div class="table-wrap">
  <table class="data-table">
    {caption_html}
    <thead><tr>{head_html}</tr></thead>
    <tbody>
{body_html}
    </tbody>
  </table>
</div>"""


def _inventory_table(rows: list[dict[str, object]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            [
                f"<td>{html.escape(str(row['dataset_label']))}</td>",
                f"<td>{_html_badge(str(row['dataset_claim_status']), _badge_class(row['dataset_claim_status']))}</td>",
                f'<td class="num">{_format_int(row["rows"])}</td>',
                f'<td class="num">{_format_int(row["rows_with_ci"])}</td>',
                f'<td class="num">{_format_int(row["headline_eligible_rows"])}</td>',
                f'<td class="num">{_format_vr(row["median_variance_ratio"])}</td>',
                f'<td class="num">{_format_pct(row["share_male_greater"])}</td>',
            ]
        )
    return _html_table(
        ["Dataset", "Claim status", "Cells", "With CI", "Headline", "Median VR", "% Male-greater"],
        html_rows,
        caption="Generated directly from the normalized cross-dataset table",
    )


def _cell_table(rows: list[dict[str, object]], *, caption: str) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            [
                f"<td>{html.escape(str(row['dataset_label']))}</td>",
                f"<td>{html.escape(_humanize_token(row['cycle_or_wave']))}</td>",
                f"<td>{html.escape(str(row['age_band']))}</td>",
                f"<td>{html.escape(str(row['trait_label']))}</td>",
                f'<td class="num">{_format_vr(row["variance_ratio"])}</td>',
                f'<td class="num">{_format_vr(row.get("vr_ci_low"))} to {_format_vr(row.get("vr_ci_high"))}</td>',
                f"<td>{_html_badge(_direction_text(row['direction']), _direction_badge_class(row['direction']))}</td>",
                f"<td>{_html_badge(str(row['claim_status_display']), _badge_class(row['claim_status_display']))}</td>",
            ]
        )
    return _html_table(
        ["Dataset", "Cycle / wave", "Age", "Trait", "VR", "95% CI", "Direction", "Claim status"],
        html_rows,
        caption=caption,
    )


def _inferential_table(rows: list[dict[str, object]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            [
                f"<td>{html.escape(str(row['dataset_label']))}</td>",
                f"<td>{html.escape(str(row['trait_label']))}</td>",
                f"<td>{html.escape(str(row['age_band']))}</td>",
                f"<td>{_html_badge(str(row['claim_status_display']), _badge_class(row['claim_status_display']))}</td>",
                f"<td>{_html_badge(_direction_text(row['direction']), _direction_badge_class(row['direction']))}</td>",
                f'<td class="num">{_format_vr(row["variance_ratio"])}</td>',
                f'<td class="num">{_format_vr(row["vr_ci_low"])} to {_format_vr(row["vr_ci_high"])}</td>',
                f'<td class="num">{_format_int(row["n_total"])}</td>',
                f"<td>{html.escape(str(row['display_explanation']))}</td>",
            ]
        )
    return _html_table(
        ["Dataset", "Trait", "Age", "Claim status", "Direction", "VR", "95% CI", "N", "Explanation"],
        html_rows,
        caption="All inferential cells from the normalized table",
    )


def _explorer_shell() -> str:
    return """<div class="callout">
  <div class="callout__title">Cell Explorer</div>
  <p>Search by dataset, trait, age, cycle, or explanation text. Then filter or sort instead of scanning multiple long ranking tables.</p>
</div>
<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin:1.25rem 0">
  <input id="explorer-search" type="search" placeholder="Search dataset, trait, age, cycle, explanation" style="flex:2 1 280px;padding:0.7rem 0.9rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);font-size:0.95rem">
  <select id="explorer-dataset" style="flex:1 1 180px;padding:0.7rem 0.9rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);font-size:0.95rem">
    <option value="">All datasets</option>
  </select>
  <select id="explorer-claim" style="flex:1 1 170px;padding:0.7rem 0.9rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);font-size:0.95rem">
    <option value="">All claim statuses</option>
    <option value="Headline claim">Headline claim</option>
    <option value="Supporting evidence">Supporting evidence</option>
    <option value="Provisional">Provisional</option>
    <option value="Method-limited">Method-limited</option>
  </select>
  <select id="explorer-direction" style="flex:1 1 150px;padding:0.7rem 0.9rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);font-size:0.95rem">
    <option value="">All directions</option>
    <option value="male_greater">Male-greater</option>
    <option value="female_greater">Female-greater</option>
    <option value="near_equal">Near equal</option>
  </select>
  <select id="explorer-sort" style="flex:1 1 170px;padding:0.7rem 0.9rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);font-size:0.95rem">
    <option value="strongest">Strongest deviations</option>
    <option value="closest">Closest to equal</option>
    <option value="largest_n">Largest N</option>
    <option value="widest_ci">Widest CI</option>
  </select>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem">
  <div id="explorer-count" style="color:var(--text-muted);font-size:0.92rem"></div>
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap">
    <button type="button" class="explorer-preset" data-preset="headline" style="padding:0.45rem 0.7rem;border:1px solid var(--border);border-radius:999px;background:var(--card-bg);color:var(--text);cursor:pointer">Headline</button>
    <button type="button" class="explorer-preset" data-preset="supporting" style="padding:0.45rem 0.7rem;border:1px solid var(--border);border-radius:999px;background:var(--card-bg);color:var(--text);cursor:pointer">Supporting</button>
    <button type="button" class="explorer-preset" data-preset="female" style="padding:0.45rem 0.7rem;border:1px solid var(--border);border-radius:999px;background:var(--card-bg);color:var(--text);cursor:pointer">Female-greater</button>
    <button type="button" class="explorer-preset" data-preset="equal" style="padding:0.45rem 0.7rem;border:1px solid var(--border);border-radius:999px;background:var(--card-bg);color:var(--text);cursor:pointer">Near equal</button>
    <button type="button" class="explorer-preset" data-preset="reset" style="padding:0.45rem 0.7rem;border:1px solid var(--border);border-radius:999px;background:var(--card-bg);color:var(--text);cursor:pointer">Reset</button>
  </div>
</div>
<div class="table-wrap" style="max-height:620px;overflow:auto">
  <table class="data-table">
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Cycle / wave</th>
        <th>Age</th>
        <th>Trait</th>
        <th>Claim</th>
        <th>Direction</th>
        <th class="num">VR</th>
        <th class="num">95% CI</th>
        <th class="num">N</th>
        <th>Explanation</th>
      </tr>
    </thead>
    <tbody id="explorer-body"></tbody>
  </table>
</div>
<div style="margin-top:0.9rem;text-align:center">
  <button id="explorer-more" type="button" style="padding:0.7rem 1rem;border:1px solid var(--border);border-radius:8px;background:var(--card-bg);color:var(--text);cursor:pointer">Load more</button>
</div>"""


def _distribution_table(rows: list[dict[str, object]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            [
                f"<td>{html.escape(str(row['dataset_label']))}</td>",
                f'<td class="num">{_format_int(row["cells"])}</td>',
                f'<td class="num">{_format_vr(row["variance_ratio_p10"])}</td>',
                f'<td class="num">{_format_vr(row["variance_ratio_p25"])}</td>',
                f'<td class="num">{_format_vr(row["variance_ratio_p50"])}</td>',
                f'<td class="num">{_format_vr(row["variance_ratio_p75"])}</td>',
                f'<td class="num">{_format_vr(row["variance_ratio_p90"])}</td>',
                f'<td class="num">{_format_pct(row["share_male_greater"])}</td>',
            ]
        )
    return _html_table(
        ["Dataset", "Cells", "P10 VR", "P25 VR", "P50 VR", "P75 VR", "P90 VR", "% Male-greater"],
        html_rows,
        caption="Per-dataset variance-ratio quantiles",
    )


def _robustness_table(rows: list[dict[str, object]], *, caption: str) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            [
                f"<td>{html.escape(str(row['dataset_label']))}</td>",
                f"<td>{html.escape(str(row['variant']))}</td>",
                f'<td class="num">{_format_int(row["matched_cells"])}</td>',
                f'<td class="num">{_format_decimal(row["median_abs_delta"])}</td>',
                f'<td class="num">{_format_pct(row["sign_change_rate"], digits=1)}</td>',
            ]
        )
    return _html_table(
        ["Dataset", "Variant", "Matched cells", "Median |Δ|", "Sign change rate"],
        html_rows,
        caption=caption,
    )


def _home_page(bundle: dict[str, object]) -> str:
    summary = bundle["summary"]
    metrics = bundle["page_metrics"]["home"]
    key_cells = bundle["key_cells"]
    strongest_positive = key_cells["headline_positive"]
    strongest_negative = key_cells["headline_counterexample"]
    body = f"""<header class="hero">
  <div class="container">
    <h1>Sex Differences in Variability Across Public-Use Datasets</h1>
    <p class="hero__subtitle">
      A single generated public bundle now drives Home, Results, Datasets, and README directly from the normalized cross-dataset table. Counts, tables, and rankings all come from the same source of truth.
    </p>
  </div>
</header>
<main>
  <section class="section">
    <div class="container">
      <h2>Headline Findings</h2>
      <p>
        The current headline layer contains {_format_int(metrics["headline_confirmatory_cell_count"])} confirmatory cells across {_format_int(metrics["live_dataset_count"])} live datasets.
        Here, confirmatory means the pre-designated core trait rows used for the main headline claim, not the secondary or exploratory rows. Male-greater variability appears in {_format_pct(metrics["headline_positive_share"])}, while the strongest counterexample remains visible rather than being averaged away.
      </p>
      <div class="stats">
        <div class="stat-card"><div class="stat-card__value">{_format_int(metrics["headline_confirmatory_cell_count"])}</div><div class="stat-card__label">Headline confirmatory cells</div></div>
        <div class="stat-card"><div class="stat-card__value">{_format_pct(metrics["headline_positive_share"])}</div><div class="stat-card__label">Male-greater within headline cells</div></div>
        <div class="stat-card"><div class="stat-card__value">{_format_int(metrics["supporting_inferential_cell_count"])}</div><div class="stat-card__label">Supporting inferential cells</div></div>
      </div>
      <p>
        The strongest male-greater confirmatory cell is <strong>{html.escape(str(strongest_positive["trait_label"]))}</strong> in <strong>{html.escape(str(strongest_positive["dataset_label"]))}</strong>,
        age <strong>{html.escape(str(strongest_positive["age_band"]))}</strong>, with VR {_format_vr(strongest_positive["variance_ratio"])}.
      </p>
    </div>
  </section>
  <section class="section">
    <div class="container">
      <div class="callout">
        <div class="callout__title">Strongest counterexample</div>
        <p>
          <strong>{html.escape(str(strongest_negative["trait_label"]))}</strong> in <strong>{html.escape(str(strongest_negative["dataset_label"]))}</strong>,
          {html.escape(_humanize_token(strongest_negative["cycle_or_wave"]))}, age <strong>{html.escape(str(strongest_negative["age_band"]))}</strong>,
          shows VR {_format_vr(strongest_negative["variance_ratio"])} with 95% CI {_format_vr(strongest_negative["vr_ci_low"])} to {_format_vr(strongest_negative["vr_ci_high"])}.
          The public summary keeps this reversal explicit.
        </p>
      </div>
    </div>
  </section>
  <section class="section">
    <div class="container">
      <h2>Headline Forest Plot</h2>
      <p>Each point is a headline-eligible confirmatory cell, meaning a core pre-designated row that is allowed into the main claim, with whiskers on the variance-ratio scale.</p>
      <div id="chart-forest" style="overflow-x:auto"></div>
    </div>
  </section>
  <section class="section">
    <div class="container">
      <h2>Dataset-Trait Medians</h2>
      <p>Median variance ratios by dataset-family combination show the distribution without relying only on the most extreme rows.</p>
      <div id="chart-hbar" style="overflow-x:auto"></div>
    </div>
  </section>
  <section class="section">
    <div class="container">
      <h2>Explore</h2>
      <div class="stats">
        <a href="results.html" class="stat-card" style="text-decoration:none;color:inherit"><div class="stat-card__value" style="font-size:1.25rem">Results</div><div class="stat-card__label">Headline, supporting, large-N, wide-CI, and near-equal views</div></a>
        <a href="datasets.html" class="stat-card" style="text-decoration:none;color:inherit"><div class="stat-card__value" style="font-size:1.25rem">Datasets</div><div class="stat-card__label">Inventory, quantiles, and age profiles from the same bundle</div></a>
        <a href="methods.html" class="stat-card" style="text-decoration:none;color:inherit"><div class="stat-card__value" style="font-size:1.25rem">Methods</div><div class="stat-card__label">Backend normalization and evidence-tier rules</div></a>
      </div>
    </div>
  </section>
</main>"""
    charts_script = f"""<script>
(function () {{
  var forestData = {json.dumps(bundle["charts"]["headline_forest"])};
  var hbarData = {json.dumps(bundle["charts"]["dataset_family_hbar"])};
  function render() {{
    charts.forest(document.getElementById('chart-forest'), forestData, {{ title: 'Headline confirmatory variance ratios' }});
    charts.hbar(document.getElementById('chart-hbar'), hbarData, {{ title: 'Median variance ratio by dataset × trait family' }});
  }}
  document.addEventListener('DOMContentLoaded', render);
}})();
</script>"""
    return _page_shell(
        title="Home",
        active="index.html",
        description="Generated public summary of sex differences in variability across public-use datasets.",
        body=body,
        charts_script=charts_script,
    )


def _results_page(bundle: dict[str, object]) -> str:
    metrics = bundle["page_metrics"]["results"]
    tables = bundle["tables"]
    body = f"""<header class="page-header">
  <div class="container">
    <h1>Results</h1>
    <p>
      This page is generated from the normalized cross-dataset table. Headline counts, supporting counts, rankings, and confidence intervals all come from the same backend bundle.
    </p>
  </div>
</header>
<main>
  <section class="section">
    <div class="container container--wide">
      <h2>Headline Confirmatory Evidence</h2>
      <p>
        There are {_format_int(metrics["headline_confirmatory_cell_count"])} headline-eligible confirmatory cells and {_format_int(metrics["supporting_inferential_cell_count"])} supporting inferential cells. Confirmatory here means the core pre-designated rows used for the main claim. The forest plot now uses variance-ratio confidence intervals rather than point estimates alone.
      </p>
      <div id="chart-forest" style="overflow-x:auto"></div>
      {_cell_table(tables["headline_confirmatory_top"], caption="Largest headline confirmatory deviations from equal variance")}
    </div>
  </section>
  <section class="section">
    <div class="container container--wide">
      <h2>Searchable Cell Explorer</h2>
      <p>The rankings still exist in the bundle, but the public page now exposes them through one searchable interface with filters and sort presets.</p>
      {_explorer_shell()}
    </div>
  </section>
  <section class="section">
    <div class="container container--wide">
      <h2>Supporting Evidence</h2>
      <p>Supporting datasets stay separate from the headline claim. Their counts and tables are generated from the same normalized rows.</p>
      {_html_table(
          ["Dataset", "Priority", "Rows", "With CI", "Headline", "Provisional", "Method-limited", "QA only", "Median VR", "% Male-greater"],
          [
              [
                  f"<td>{html.escape(str(row['dataset_label']))}</td>",
                  f"<td>{html.escape(str(row['trait_priority']))}</td>",
                  f'<td class="num">{_format_int(row["rows"])}</td>',
                  f'<td class="num">{_format_int(row["rows_with_ci"])}</td>',
                  f'<td class="num">{_format_int(row["headline_eligible_rows"])}</td>',
                  f'<td class="num">{_format_int(row["provisional_rows"])}</td>',
                  f'<td class="num">{_format_int(row["method_limited_rows"])}</td>',
                  f'<td class="num">{_format_int(row["qa_only_rows"])}</td>',
                  f'<td class="num">{_format_vr(np.exp(row["median_log_variance_ratio"]))}</td>',
                  f'<td class="num">{_format_pct(row["share_male_greater"])}</td>',
              ]
              for row in tables["supporting_summary"]
          ],
          caption="Supporting-evidence summary"
      )}
      {_cell_table(tables["supporting_top_cells"], caption="Largest supporting-evidence cells")}
    </div>
  </section>
  <section class="section">
    <div class="container container--wide">
      <h2>Robustness</h2>
      <p>Robustness summaries remain build artifacts, but the public tables and counts here are now generated from the same bundle.</p>
      {_robustness_table(tables["robustness_summary"], caption="Cross-dataset robustness summary") if tables["robustness_summary"] else ""}
      <div id="chart-robustness" style="overflow-x:auto"></div>
    </div>
  </section>
</main>"""
    charts_script = f"""<script>
(function () {{
  var forestData = {json.dumps(bundle["charts"]["headline_forest"])};
  var robustnessData = {json.dumps(bundle["charts"]["robustness"])};
  var explorerRows = {json.dumps(tables["explorer_cells"])};
  var visibleRows = 24;

  function humanizeCycle(value) {{
    return String(value || '—').replaceAll('_', ' ').replace(/\\b\\w/g, function (m) {{ return m.toUpperCase(); }});
  }}

  function badgeClassForClaim(value) {{
    return {{
      'Headline claim': 'badge--headline',
      'Supporting evidence': 'badge--supporting',
      'Provisional': 'badge--provisional',
      'Method-limited': 'badge--method',
      'QA only': 'badge--qa'
    }}[value] || 'badge--qa';
  }}

  function badgeClassForDirection(value) {{
    return {{
      'male_greater': 'badge--headline',
      'female_greater': 'badge--supporting',
      'near_equal': 'badge--qa'
    }}[value] || 'badge--qa';
  }}

  function directionText(value) {{
    return {{
      'male_greater': 'M > F',
      'female_greater': 'F > M',
      'near_equal': 'Near equal'
    }}[value] || 'Unavailable';
  }}

  function fmtVR(value) {{
    return typeof value === 'number' ? value.toFixed(2) + 'x' : '—';
  }}

  function fmtInt(value) {{
    return typeof value === 'number' ? value.toLocaleString() : '—';
  }}

  function populateDatasets() {{
    var select = document.getElementById('explorer-dataset');
    var datasets = Array.from(new Set(explorerRows.map(function (row) {{ return row.dataset_label; }}))).sort();
    datasets.forEach(function (label) {{
      var option = document.createElement('option');
      option.value = label;
      option.textContent = label;
      select.appendChild(option);
    }});
  }}

  function filteredRows() {{
    var search = document.getElementById('explorer-search').value.trim().toLowerCase();
    var dataset = document.getElementById('explorer-dataset').value;
    var claim = document.getElementById('explorer-claim').value;
    var direction = document.getElementById('explorer-direction').value;
    var sort = document.getElementById('explorer-sort').value;

    var rows = explorerRows.filter(function (row) {{
      var haystack = [
        row.dataset_label,
        row.trait_label,
        row.age_band,
        row.cycle_or_wave,
        row.display_explanation
      ].join(' ').toLowerCase();
      return (!search || haystack.indexOf(search) !== -1)
        && (!dataset || row.dataset_label === dataset)
        && (!claim || row.claim_status_display === claim)
        && (!direction || row.direction === direction);
    }});

    rows.sort(function (a, b) {{
      if (sort === 'closest') {{
        return (a.distance_from_equal || 0) - (b.distance_from_equal || 0);
      }}
      if (sort === 'largest_n') {{
        return (b.n_total || 0) - (a.n_total || 0);
      }}
      if (sort === 'widest_ci') {{
        return (b.ci_width || 0) - (a.ci_width || 0);
      }}
      return (b.abs_log_vr || 0) - (a.abs_log_vr || 0);
    }});
    return rows;
  }}

  function renderExplorer() {{
    var rows = filteredRows();
    var tbody = document.getElementById('explorer-body');
    var count = document.getElementById('explorer-count');
    var more = document.getElementById('explorer-more');
    tbody.innerHTML = '';
    rows.slice(0, visibleRows).forEach(function (row) {{
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + row.dataset_label + '</td>' +
        '<td>' + humanizeCycle(row.cycle_or_wave) + '</td>' +
        '<td>' + row.age_band + '</td>' +
        '<td>' + row.trait_label + '</td>' +
        '<td><span class="badge ' + badgeClassForClaim(row.claim_status_display) + '">' + row.claim_status_display + '</span></td>' +
        '<td><span class="badge ' + badgeClassForDirection(row.direction) + '">' + directionText(row.direction) + '</span></td>' +
        '<td class="num">' + fmtVR(row.variance_ratio) + '</td>' +
        '<td class="num">' + fmtVR(row.vr_ci_low) + ' to ' + fmtVR(row.vr_ci_high) + '</td>' +
        '<td class="num">' + fmtInt(row.n_total) + '</td>' +
        '<td>' + row.display_explanation + '</td>';
      tbody.appendChild(tr);
    }});
    count.textContent = rows.length + ' matching cells';
    more.style.display = rows.length > visibleRows ? 'inline-flex' : 'none';
  }}

  function attachExplorer() {{
    populateDatasets();
    ['explorer-search', 'explorer-dataset', 'explorer-claim', 'explorer-direction', 'explorer-sort'].forEach(function (id) {{
      document.getElementById(id).addEventListener('input', function () {{
        visibleRows = 24;
        renderExplorer();
      }});
      document.getElementById(id).addEventListener('change', function () {{
        visibleRows = 24;
        renderExplorer();
      }});
    }});
    document.getElementById('explorer-more').addEventListener('click', function () {{
      visibleRows += 24;
      renderExplorer();
    }});
    document.querySelectorAll('.explorer-preset').forEach(function (button) {{
      button.addEventListener('click', function () {{
        var preset = button.getAttribute('data-preset');
        if (preset === 'headline') {{
          document.getElementById('explorer-claim').value = 'Headline claim';
          document.getElementById('explorer-direction').value = '';
          document.getElementById('explorer-sort').value = 'strongest';
        }} else if (preset === 'supporting') {{
          document.getElementById('explorer-claim').value = 'Supporting evidence';
          document.getElementById('explorer-direction').value = '';
          document.getElementById('explorer-sort').value = 'strongest';
        }} else if (preset === 'female') {{
          document.getElementById('explorer-direction').value = 'female_greater';
          document.getElementById('explorer-sort').value = 'strongest';
        }} else if (preset === 'equal') {{
          document.getElementById('explorer-direction').value = 'near_equal';
          document.getElementById('explorer-sort').value = 'closest';
        }} else {{
          document.getElementById('explorer-search').value = '';
          document.getElementById('explorer-dataset').value = '';
          document.getElementById('explorer-claim').value = '';
          document.getElementById('explorer-direction').value = '';
          document.getElementById('explorer-sort').value = 'strongest';
        }}
        visibleRows = 24;
        renderExplorer();
      }});
    }});
    renderExplorer();
  }}

  function render() {{
    charts.forest(document.getElementById('chart-forest'), forestData, {{ title: 'Headline confirmatory variance ratios' }});
    if (robustnessData.length) {{
      charts.robustness(document.getElementById('chart-robustness'), robustnessData);
    }}
    attachExplorer();
  }}
  document.addEventListener('DOMContentLoaded', render);
}})();
</script>"""
    return _page_shell(
        title="Results",
        active="results.html",
        description="Generated results page for sex variability analyses.",
        body=body,
        charts_script=charts_script,
    )


def _datasets_page(bundle: dict[str, object]) -> str:
    tables = bundle["tables"]
    body = f"""<header class="page-header">
  <div class="container">
    <h1>Datasets</h1>
    <p>
      The inventory, quantiles, and age profiles on this page are generated from the same normalized table as the rest of the public site.
    </p>
  </div>
</header>
<main>
  <section class="section">
    <div class="container container--wide">
      <h2>Dataset Inventory</h2>
      {_inventory_table(tables["dataset_inventory"])}
    </div>
  </section>
  <section class="section">
    <div class="container container--wide">
      <h2>Per-Dataset Quantiles</h2>
      <p>These quantiles provide distribution views without requiring new data collection.</p>
      {_distribution_table(tables["dataset_distribution"])}
    </div>
  </section>
  <section class="section">
    <div class="container container--wide">
      <h2>Age Profiles</h2>
      <p>Median log variance ratios across age or grade bands within each dataset.</p>
      <div id="chart-age" style="overflow-x:auto"></div>
    </div>
  </section>
</main>"""
    charts_script = f"""<script>
(function () {{
  var ageData = {json.dumps(bundle["charts"]["age_profile"])};
  function render() {{
    charts.ageProfile(document.getElementById('chart-age'), ageData);
  }}
  document.addEventListener('DOMContentLoaded', render);
}})();
</script>"""
    return _page_shell(
        title="Datasets",
        active="datasets.html",
        description="Generated dataset inventory for sex variability analyses.",
        body=body,
        charts_script=charts_script,
    )


def render_site_pages(bundle: dict[str, object], site_dir: Path) -> dict[str, Path]:
    validate_site_bundle(bundle)
    site_dir.mkdir(parents=True, exist_ok=True)
    pages = {
        "index.html": _home_page(bundle),
        "results.html": _results_page(bundle),
        "datasets.html": _datasets_page(bundle),
    }
    written: dict[str, Path] = {}
    for name, content in pages.items():
        path = site_dir / name
        path.write_text(content, encoding="utf-8")
        written[name] = path
    return written


def render_readme(bundle: dict[str, object], output_path: Path) -> Path:
    validate_site_bundle(bundle)
    summary = bundle["summary"]
    tables = bundle["tables"]
    strongest_positive = bundle["key_cells"]["headline_positive"]
    strongest_negative = bundle["key_cells"]["headline_counterexample"]

    headline_table = pd.DataFrame(
        [
            {
                "Metric": "Headline-eligible confirmatory cells",
                "Value": int(summary["headline_confirmatory_cell_count"]),
            },
            {
                "Metric": "Share male-greater",
                "Value": _format_pct(summary["headline_positive_share"]),
            },
            {
                "Metric": "Median variance ratio",
                "Value": _format_vr(summary["headline_median_variance_ratio"]),
            },
            {
                "Metric": "Mean variance ratio",
                "Value": _format_vr(summary["headline_mean_variance_ratio"]),
            },
            {
                "Metric": "Range",
                "Value": f"{_format_vr(summary['headline_min_variance_ratio'])} to {_format_vr(summary['headline_max_variance_ratio'])}",
            },
            {
                "Metric": "Datasets contributing",
                "Value": int(summary["headline_dataset_count"]),
            },
        ]
    )
    inventory_df = pd.DataFrame(tables["dataset_inventory"])[
        ["dataset_label", "dataset_claim_status", "rows", "rows_with_ci", "headline_eligible_rows", "median_variance_ratio", "share_male_greater"]
    ].rename(
        columns={
            "dataset_label": "Dataset",
            "dataset_claim_status": "Claim status",
            "rows": "Cells",
            "rows_with_ci": "With CI",
            "headline_eligible_rows": "Headline rows",
            "median_variance_ratio": "Median VR",
            "share_male_greater": "% Male-greater",
        }
    )
    inventory_df["Median VR"] = inventory_df["Median VR"].map(_format_vr)
    inventory_df["% Male-greater"] = inventory_df["% Male-greater"].map(_format_pct)
    selected_cells_df = pd.DataFrame(tables["headline_confirmatory_top"]).head(8)[
        ["dataset_label", "trait_label", "age_band", "variance_ratio", "vr_ci_low", "vr_ci_high", "claim_status_display"]
    ].rename(
        columns={
            "dataset_label": "Dataset",
            "trait_label": "Trait",
            "age_band": "Age",
            "variance_ratio": "VR",
            "vr_ci_low": "CI low",
            "vr_ci_high": "CI high",
            "claim_status_display": "Claim status",
        }
    )
    for column in ["VR", "CI low", "CI high"]:
        selected_cells_df[column] = selected_cells_df[column].map(_format_vr)

    lines = [
        "# Sex Differences in Variability Across Public-Use Datasets",
        "",
        "**[Interactive results site](https://smkwray.github.io/sexvary/)** · "
        "[Results](https://smkwray.github.io/sexvary/results.html) · "
        "[Datasets](https://smkwray.github.io/sexvary/datasets.html) · "
        "[Methods](https://smkwray.github.io/sexvary/methods.html) · "
        "[Limits](https://smkwray.github.io/sexvary/limits.html) · "
        "[Explanations](https://smkwray.github.io/sexvary/explanations.html)",
        "",
        "---",
        "",
        f"This project estimates where sex differences in score variability appear across **{_format_int(summary['live_dataset_count'])} live datasets**. The public README is generated from the same normalized bundle as the site pages, so counts and tables stay aligned.",
        "",
        "The current public bundle includes NIH Collaborative Perinatal Project outputs for core cognition and growth trajectories. These CPP rows are visible in the dataset inventory and cell explorer, and currently remain method-limited rather than headline-eligible.",
        "",
        "## Headline findings",
        "",
        markdown_table(headline_table),
        "",
        f"- Strongest positive: **{strongest_positive['trait_label']}** in **{strongest_positive['dataset_label']}**, age **{strongest_positive['age_band']}** (VR {_format_vr(strongest_positive['variance_ratio'])})",
        f"- Strongest counterexample: **{strongest_negative['trait_label']}** in **{strongest_negative['dataset_label']}**, age **{strongest_negative['age_band']}** (VR {_format_vr(strongest_negative['variance_ratio'])})",
        f"- Supporting evidence: **{_format_int(summary['supporting_inferential_cell_count'])}** inferential rows from NHANES, HRS, and PSID remain separate from the headline claim",
        "",
        "## Selected headline cells",
        "",
        markdown_table(selected_cells_df),
        "",
        "## Datasets",
        "",
        markdown_table(inventory_df),
        "",
        "## Distribution views",
        "",
        "The generated bundle now exports strongest male-greater rows, strongest female-greater rows, closest-to-equal rows, largest-N rows, widest-CI rows, and per-dataset variance-ratio quantiles and histograms.",
        "",
        "## Methods",
        "",
        "The pipeline computes sex-specific weighted variances within dataset-defined cells. Public counts and display tables are generated from the normalized cross-dataset table rather than hand-maintained page content.",
        "",
        "## Reproducibility",
        "",
        "Run `python scripts/run_paper_bundle.py` after the backend comparison build to regenerate the site bundle, public pages, and README together.",
        "",
        "## License",
        "",
        "MIT",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

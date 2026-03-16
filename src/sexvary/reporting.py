from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        header = "| " + " | ".join(str(col) for col in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *rows])


def write_markdown_summary(df: pd.DataFrame, path: str | Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append(markdown_table(df))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def forest_plot_from_effects(
    df: pd.DataFrame,
    *,
    label_col: str,
    effect_col: str,
    se_col: str,
    output_path: str | Path,
    title: str = "Forest plot",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    work = df.copy().reset_index(drop=True)
    y = np.arange(len(work))[::-1]
    lo = work[effect_col] - 1.96 * work[se_col]
    hi = work[effect_col] + 1.96 * work[se_col]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(work) + 1.5)))
    ax.errorbar(work[effect_col], y, xerr=[work[effect_col] - lo, hi - work[effect_col]], fmt="o")
    ax.axvline(0.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(work[label_col])
    ax.set_xlabel("Effect size")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

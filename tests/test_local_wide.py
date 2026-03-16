from pathlib import Path

import pandas as pd

from sexvary.adapters import LocalWideTableAdapter
from sexvary.config import build_registry
from sexvary.io import write_table


def test_local_wide_adapter_uses_weight_fallback_columns(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": [1, 2],
            "csage": [12.0, 12.5],
            "child_sampling_weight_1998": [0.0, 0.0],
            "child_sampling_weight_2016_under14": [2.5, 3.5],
            "PPVT": [100.0, 90.0],
        }
    )
    raw_path = write_table(raw, tmp_path / "child.csv")
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "dataset_id: nlsy79_child_ya",
                "table_shape: wide",
                "columns:",
                "  person_id: person_id",
                "  sex: sex",
                "  age: csage",
                "  weight: child_sampling_weight_1998",
                "weight_fallback_columns:",
                "  - child_sampling_weight_1998",
                "  - child_sampling_weight_2016_under14",
                "traits:",
                "  ppvt: PPVT",
                "value_maps:",
                "  sex:",
                '    male: [1, "1"]',
                '    female: [2, "2"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    adapter = LocalWideTableAdapter(build_registry().get_dataset("nlsy79_child_ya"), raw_path=raw_path, mapping_path=mapping_path)
    out = adapter.to_long_person_trait().data.sort_values("person_id", kind="stable").reset_index(drop=True)
    assert out["weight_main"].tolist() == [2.5, 3.5]
    assert out["weight_source"].tolist() == ["child_sampling_weight_2016_under14", "child_sampling_weight_2016_under14"]
    assert out["weight_primary_source"].tolist() == ["child_sampling_weight_1998", "child_sampling_weight_1998"]

# Contributing

Developer setup and pipeline reference for the sexvary analysis backend.

## Setup

```bash
# Create environment
uv venv "$HOME/venvs/sexvary" --python 3.11
uv pip install --python "$HOME/venvs/sexvary/bin/python" -e .[dev]

# Run tests
make test

# Validate project layout
make validate
```

## Register local NLSY extracts

Copy the example config and edit with your paths:

```bash
cp config/local_paths.example.yaml config/local_paths.yaml
```

Then register each dataset:

```bash
"$HOME/venvs/sexvary/bin/python" scripts/register_nlsy.py --dataset-id nlsy79_main --path /path/to/nlsy79_cfa.csv
"$HOME/venvs/sexvary/bin/python" scripts/register_nlsy.py --dataset-id nlsy97_main --path /path/to/nlsy97_cfa.csv
"$HOME/venvs/sexvary/bin/python" scripts/register_nlsy.py --dataset-id nlsy79_child_ya --path /path/to/cnlsy_cfa.csv
```

Relative paths in `config/local_paths.yaml` are resolved from the project root.

External public-use datasets resolve from the canonical shared data root first (`../data/sources/{provider}/{dataset}/{version}/...`), then fall back to legacy repo-local paths. Set `PROJ_SHARED_DATA_ROOT` to override.

## Pipeline commands

| Command | Description |
| --- | --- |
| `make local-nlsy` | Run local NLSY estimate pipeline |
| `make piaac` | Run PIAAC cycle 2 pipeline |
| `make pisa` | Run PISA 2022 pipeline |
| `make timss` | Run TIMSS pipeline (default: 2019) |
| `make timss DATASET_ID=timss_2023` | Run TIMSS 2023 |
| `make pirls` | Run PIRLS 2021 pipeline |
| `make icils` | Run ICILS 2023 pipeline |
| `make nhanes` | Run NHANES selected-cycles pipeline |
| `make nnyfs` | Run NNYFS 2012 pipeline |
| `make psid` | Run PSID CDS / TAS pipeline |
| `make hrs` | Run HRS public pipeline |
| `make nces-school DATASET_ID=ecls_k_2011` | Run ECLS-K:2011 pipeline |
| `make nces-school DATASET_ID=hsls_2009` | Run HSLS:09 pipeline |
| `make compare-results` | Build cross-dataset comparison tables and figures |
| `make paper-report` | Build paper-style report bundle |
| `make backend` | Run all available pipelines end to end |
| `make backend-health` | Build local-vs-remote health report |
| `make update-site-assets` | Copy fresh figures to site directory |

## Full backend rebuild

```bash
make backend
```

This detects which pipelines are runnable from available data, runs them in sequence, and rebuilds the cross-dataset comparison layer. Use `--dry-run` to preview.

## Repository layout

```text
sexvary/
├── config/          # Dataset, trait, and analysis configuration
├── data/            # Raw input data (gitignored)
├── results/         # Pipeline outputs (gitignored, regenerated)
├── scripts/         # Pipeline runner scripts
├── site/            # GitHub Pages static site
├── src/sexvary/     # Main Python package
└── tests/           # Test suite
```

## Evidence status system

Every output cell carries a label:

- **headline-eligible** — confirmatory trait, design-aware inference, no caveats
- **supporting** — secondary/exploratory priority
- **provisional** — relies on fallback weights
- **method-limited** — simple-design SE only
- **QA-only** — below thresholds or bounded-scale issues

## Defensibility rules

- Keep the project **descriptive**
- Do not turn observed score variance into a claim about innate capacity
- Report **male-greater, female-greater, and null** variance patterns
- Separate **instrument / age / cohort / population** effects
- Document exclusions, topcoding, bounded scales, and missing-data choices
- Never bypass terms of use or public-data restrictions

## License

MIT

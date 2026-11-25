# Contributing

Developer setup and pipeline reference for the sexvary analysis backend.

## Setup

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

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
python scripts/register_nlsy.py --dataset-id nlsy79_main --path /path/to/nlsy79_cfa.csv
python scripts/register_nlsy.py --dataset-id nlsy97_main --path /path/to/nlsy97_cfa.csv
python scripts/register_nlsy.py --dataset-id nlsy79_child_ya --path /path/to/cnlsy_cfa.csv
```

Relative paths in `config/local_paths.yaml` are resolved from the project root.

External public-use datasets are read from `<data root>/sources/{provider}/{dataset}/{version}/`, then from `data/raw/external/{dataset_id}/`. The data root is `PROJ_SHARED_DATA_ROOT` if set, otherwise `../data` if that folder exists, otherwise `data/`.

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
| `make update-site-assets` | Copy fresh figures to site directory |

## Full backend rebuild

```bash
make backend
```

This detects which pipelines are runnable from available data, runs them in sequence, and rebuilds the cross-dataset comparison layer. To preview without running, use `python scripts/run_backend.py --dry-run`.

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

## License

MIT

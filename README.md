# Sex Differences in Variability Across Public-Use Datasets

**[Interactive results site](https://smkwray.github.io/sexvary/)** · [Results](https://smkwray.github.io/sexvary/results.html) · [Datasets](https://smkwray.github.io/sexvary/datasets.html) · [Methods](https://smkwray.github.io/sexvary/methods.html) · [Limits](https://smkwray.github.io/sexvary/limits.html) · [Explanations](https://smkwray.github.io/sexvary/explanations.html)

---

Are test and measurement scores more spread out among males or among females, and does the answer change with the trait, the age group, and the dataset? This project estimates sex differences in score variability across 17 public-use datasets. The main measure is the variance ratio (VR), male variance divided by female variance: a VR above 1 means male scores vary more, and a VR below 1 means female scores vary more.

The analysis is descriptive. Observed score variance on public-use instruments is not a measure of innate or biological variability.

## Claim-status glossary

<details>
<summary><strong>What "Headline claim", "Supporting evidence", "Provisional", and "Method-limited" mean</strong></summary>

- **Headline claim**: Core confirmatory rows with the strongest survey-design handling and no fallback flags. Only these rows feed the main headline claim.
- **Supporting evidence**: Inferential rows kept for added age or domain coverage, but still kept separate from the main headline claim.
- **Provisional**: Rows that depend on fallback weights or other temporary inference paths. They stay visible, but they do not count as headline evidence.
- **Method-limited**: Rows with usable point estimates but weaker uncertainty estimates, usually because only simple effective-sample-size approximations were available.
- **QA only**: Rows retained for transparency and diagnostics only, not for inferential claims.

</details>

## Headline findings

| Metric | Value |
| --- | --- |
| Headline-eligible confirmatory cells | 67 |
| Share male-greater | 85% |
| Median variance ratio | 1.15x |
| Mean variance ratio | 1.14x |
| Range | 0.10x to 1.46x |
| Datasets contributing | 7 |

- Strongest positive: Numeracy in PIAAC cycle 2, age 60-65 (VR 1.46x)
- Strongest counterexample: Reading achievement in ECLS-K:2011, age K (VR 0.10x)
- Supporting evidence: 263 inferential rows from NHANES, HRS, and PSID are reported separately from the headline claim

## Selected headline cells

| Dataset | Trait | Age | VR | CI low | CI high | Claim status |
| --- | --- | --- | --- | --- | --- | --- |
| ECLS-K:2011 | Reading achievement | K | 0.10x | 0.10x | 0.10x | Headline claim |
| PIAAC cycle 2 | Numeracy | 60-65 | 1.46x | 1.21x | 1.76x | Headline claim |
| PIAAC cycle 2 | Numeracy | 30-34 | 1.44x | 1.13x | 1.84x | Headline claim |
| PIAAC cycle 2 | Adaptive problem solving | 45-49 | 1.41x | 1.02x | 1.95x | Headline claim |
| PIAAC cycle 2 | Numeracy | 45-49 | 1.38x | 1.00x | 1.91x | Headline claim |
| PIAAC cycle 2 | Adaptive problem solving | 35-39 | 1.38x | 0.99x | 1.93x | Headline claim |
| PIAAC cycle 2 | Literacy | 60-65 | 1.38x | 1.13x | 1.69x | Headline claim |
| ECLS-K:2011 | Reading achievement | 4 | 1.37x | 1.25x | 1.50x | Headline claim |

## Datasets

| Dataset | Claim status | Cells | With CI | Eligible rows | Median VR | % Male-greater |
| --- | --- | --- | --- | --- | --- | --- |
| CPP core cognition | Method-limited | 4 | 4 | 0 | 1.00x | 50% |
| CPP growth trajectories | Method-limited | 15 | 15 | 0 | 1.01x | 53% |
| ECLS-K:2011 | Headline claim | 32 | 26 | 23 | 1.18x | 85% |
| HRS public | Provisional | 182 | 91 | 0 | 0.97x | 40% |
| HSLS:09 | Headline claim | 2 | 2 | 2 | 1.14x | 100% |
| ICILS 2023 | Supporting evidence | 2 | 2 | 2 | 1.21x | 100% |
| NHANES selected cycles | Supporting evidence | 472 | 132 | 132 | 1.04x | 58% |
| NLSY79 Child and Young Adult | Provisional | 23 | 5 | 0 | 1.38x | 100% |
| NLSY79 main | Method-limited | 18 | 18 | 0 | 1.23x | 89% |
| NLSY97 main | Method-limited | 11 | 11 | 0 | 1.28x | 100% |
| NNYFS 2012 | Supporting evidence | 19 | 14 | 14 | 1.01x | 50% |
| PIAAC cycle 2 | Headline claim | 30 | 30 | 30 | 1.14x | 87% |
| PIRLS 2021 | Headline claim | 3 | 3 | 3 | 0.99x | 33% |
| PISA 2022 | Headline claim | 3 | 3 | 3 | 1.25x | 100% |
| PSID CDS / TAS | Provisional | 112 | 40 | 0 | 1.10x | 68% |
| TIMSS 2019 | Headline claim | 4 | 4 | 4 | 1.12x | 100% |
| TIMSS 2023 | Headline claim | 4 | 4 | 4 | 1.13x | 100% |

Eligible rows have a confidence interval and no provisional or method-limited flag. Only eligible rows on confirmatory traits count as headline cells (67 across 7 datasets), so eligible rows in supporting-evidence datasets stay outside the headline claim.

## Distribution views

The generated bundle also exports strongest male-greater rows, strongest female-greater rows, closest-to-equal rows, largest-N rows, widest-CI rows, and per-dataset variance-ratio quantiles and histograms.

## Methods

The pipeline computes sex-specific weighted variances within dataset-defined cells (a trait at an age band and wave within one dataset) and reports their ratio, with a confidence interval where the data support one. The README and the site pages are generated from the same normalized cross-dataset table, so their counts and tables match. See [Methods](https://smkwray.github.io/sexvary/methods.html) for details.

## Reproducibility

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest

python scripts/run_backend.py        # run every pipeline whose input data are present, then rebuild the comparison
python scripts/run_paper_bundle.py   # regenerate the site bundle, site pages, and this README
```

Raw microdata are not included. `python scripts/acquire_public_data.py` creates a folder for each external dataset with a `SOURCE.md` note giving its official URL and access mode. Some datasets require registration, and their terms of use prohibit redistributing the raw files.

## Limitations

- Results describe observed scores on specific public-use instruments, ages, and cohorts; a result on one measure, population, or age does not establish the pattern for others.
- Public-use files mask some survey design variables and may suppress small cells in the tails, which limits design-aware inference for variability.
- Bounded scales distort variance. Kindergarten reading scores in ECLS-K:2011 show floor effects at school entry: the fall-kindergarten reading cell shows much greater female variability, while the same sample shows greater male variability in reading by later grades.

See [Limits](https://smkwray.github.io/sexvary/limits.html) for the full list.

## License

[MIT](LICENSE)

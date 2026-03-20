# Sex Differences in Variability Across Public-Use Datasets

**[Interactive results site](https://smkwray.github.io/sexvary/)** · [Results](https://smkwray.github.io/sexvary/results.html) · [Datasets](https://smkwray.github.io/sexvary/datasets.html) · [Methods](https://smkwray.github.io/sexvary/methods.html) · [Limits](https://smkwray.github.io/sexvary/limits.html) · [Explanations](https://smkwray.github.io/sexvary/explanations.html)

---

This project estimates where sex differences in score variability appear across **15 live datasets**. The public README is generated from the same normalized bundle as the site pages, so counts and tables stay aligned.

## Headline findings

| Metric | Value |
| --- | --- |
| Headline-eligible confirmatory cells | 67 |
| Share male-greater | 85% |
| Median variance ratio | 1.15x |
| Mean variance ratio | 1.14x |
| Range | 0.10x to 1.46x |
| Datasets contributing | 7 |

- Strongest positive: **Numeracy** in **PIAAC cycle 2**, age **60-65** (VR 1.46x)
- Strongest counterexample: **Reading achievement** in **ECLS-K:2011**, age **K** (VR 0.10x)
- Supporting evidence: **263** inferential rows from NHANES, HRS, and PSID remain separate from the headline claim

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

| Dataset | Claim status | Cells | With CI | Headline rows | Median VR | % Male-greater |
| --- | --- | --- | --- | --- | --- | --- |
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

## Distribution views

The generated bundle now exports strongest male-greater rows, strongest female-greater rows, closest-to-equal rows, largest-N rows, widest-CI rows, and per-dataset variance-ratio quantiles and histograms.

## Methods

The pipeline computes sex-specific weighted variances within dataset-defined cells. Public counts and display tables are generated from the normalized cross-dataset table rather than hand-maintained page content.

## Reproducibility

Run `python scripts/run_paper_bundle.py` after the backend comparison build to regenerate the site bundle, public pages, and README together.

## License

MIT

# Sex Differences in Variability Across Public-Use Datasets

**[Interactive results site](https://smkwray.github.io/sexvary/)** ·
[Results](https://smkwray.github.io/sexvary/results.html) ·
[Datasets](https://smkwray.github.io/sexvary/datasets.html) ·
[Methods](https://smkwray.github.io/sexvary/methods.html) ·
[Limits](https://smkwray.github.io/sexvary/limits.html) ·
[Explanations](https://smkwray.github.io/sexvary/explanations.html)

---

This project estimates where sex differences in score variability appear — and where they do not — across cognitive, achievement, and physical traits, using **14 public-use datasets** and over **424,000 observations**. The core metric is the **variance ratio** (male variance / female variance). A VR above 1.0× means males are more variable; below 1.0× means females are. The analysis is descriptive, not causal.

## Headline findings

Across **50 headline-eligible confirmatory cells** from 7 datasets with design-aware inference (plausible values, BRR, JRR, or stratified bootstrap), about **94% show male-greater variability**.

| Metric | Value |
| --- | --- |
| Headline-eligible cells | 50 |
| Share male-greater | 94% |
| Median variance ratio | 1.18× |
| Mean variance ratio | 1.16× |
| Range | 0.10× to 1.46× |
| Datasets contributing | 7 |
| Total observations (headline) | 424,066 |

- **Strongest positive:** adult numeracy in PIAAC cycle 2, ages 60–65 (VR 1.46× — males 46% more variable)
- **Strongest counterexample:** kindergarten reading in ECLS-K:2011 (VR 0.10× — females far more variable at school entry)
- **Supporting evidence:** 172 additional inferential rows from NHANES, PSID, and HRS broaden age and domain coverage but remain below headline quality

### Selected cells with sex-specific statistics

Where the pipeline stores individual sex means and variances, these are available per cell. PV-based datasets (PIAAC, PISA, TIMSS) report the mean difference across plausible-value draws. The full table with all 294 inferential cells is available in the [interactive cell explorer](https://smkwray.github.io/sexvary/results.html).

| Dataset | Trait | Age | Male Mean | Female Mean | Male Var | Female Var | VR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ECLS-K:2011 | Reading | K (fall) | −0.80 | 1.77 | 18,241 | 179,752 | 0.10× |
| ECLS-K:2011 | Reading | 4th (spring) | 0.52 | 0.38 | 0.54 | 0.40 | 1.37× |
| ECLS-K:2011 | Math | K (fall) | −1.15 | −1.15 | 0.54 | 0.44 | 1.21× |
| NHANES | Grip strength | 12–15 | 33.1 kg | 26.9 kg | 82.9 | 24.5 | 3.39× |
| NHANES | Height | 12–15 | 166.2 cm | 159.3 cm | 93.2 | 39.2 | 2.38× |
| NHANES | BMI | 0–3 | 16.5 | 16.3 | 2.4 | 3.5 | 0.68× |
| PIAAC | Numeracy | 60–65 | — | — | — | — | 1.46× |
| PISA 2022 | Math | 15-yr | — | — | — | — | 1.30× |

*Male Mean / Female Mean = weighted group means. Male Var / Female Var = weighted group variances. PV datasets show — for individual means (mean difference available in the cell explorer). VR = Var_male / Var_female.*

## Datasets

| Dataset | Evidence tier | Cells | Combined N | Median VR | % Male-greater |
| --- | --- | --- | --- | --- | --- |
| ECLS-K:2011 | headline | 23 | 182,376 | 1.18× | 85% |
| PIAAC cycle 2 | headline | 30 | 11,295 | 1.14× | 87% |
| PISA 2022 | headline | 3 | 13,641 | 1.25× | 100% |
| TIMSS 2019 | headline | 4 | 46,200 | 1.12× | 100% |
| TIMSS 2023 | headline | 4 | 34,320 | 1.13× | 100% |
| PIRLS 2021 | headline | 3 | 4,968 | 0.99× | 33% |
| HSLS:09 | headline | 2 | 37,246 | 1.14× | 100% |
| ICILS 2023 | headline | 2 | 4,642 | 1.21× | 100% |
| NHANES selected cycles | supporting | 132 | 82,600 | 1.04× | 58% |
| NNYFS 2012 | supporting | 14 | 6,778 | 1.01× | 50% |
| PSID CDS / TAS | supporting | 40 | — | 1.10× | 68% |
| NLSY79 main | method-limited | 18 | — | 1.23× | 89% |
| NLSY97 main | method-limited | 11 | — | 1.28× | 100% |
| NLSY79 Child/YA | provisional | 5 | — | 1.38× | 100% |

All 14 datasets span U.S. populations from kindergarten through older adulthood, covering achievement, cognition, adult skills, physical traits, and digital literacy.

## Traits covered

The project analyzes 15 distinct traits across these families:

- **Achievement:** math, reading, and science achievement (ECLS-K, PISA, TIMSS, PIRLS, HSLS)
- **Adult skills:** literacy, numeracy, and adaptive problem solving (PIAAC)
- **Digital skills:** computer/information literacy, computational thinking (ICILS)
- **Cognition:** ASVAB subtests, PPVT, PIAT, digit span (NLSY family)
- **Physical:** height, weight, BMI, waist circumference, grip strength (NHANES, NNYFS, PSID)

## Methods

The pipeline computes sex-specific weighted variances within analysis cells defined by dataset, cycle/wave, country, age/grade band, and trait. Inference is matched to each dataset's survey design:

| Inference method | Datasets |
| --- | --- |
| Plausible values + replicate weights | PIAAC, PISA, TIMSS, PIRLS, ICILS |
| BRR replicate weights | HSLS:09 |
| Stratified PSU bootstrap | ECLS-K:2011, NHANES, NNYFS |
| Simple-design SE approximation | NLSY79, NLSY97, NLSY79 Child/YA, PSID |

Every cell carries an **evidence-status label** — headline-eligible, supporting, provisional, method-limited, or QA-only — so that interpretation does not depend on reading code.

## Evidence tiers

| Tier | Meaning | Count |
| --- | --- | --- |
| **Headline-eligible** | Confirmatory trait, design-aware inference, no caveats | 50 cells |
| **Supporting** | Secondary/exploratory priority, broadens domain coverage | 132 cells |
| **Provisional** | Relies on fallback weights or alternate inference paths | 11 cells |
| **Method-limited** | Simple-design SE only, no replicate weights | 77 cells |
| **QA-only** | Below thresholds or bounded-scale issues | 441 cells |

<details>
<summary><strong>Why might males show greater variability? (X-linked hypothesis)</strong></summary>

One biologically plausible **partial** explanation is the asymmetric genetics of the X chromosome. Males are hemizygous for most non-pseudoautosomal X-linked loci, so X-linked allelic effects are more directly exposed. Females have two X chromosomes, but one is largely inactivated in each cell, producing mosaic expression. Under standard dosage-compensation models, that architecture predicts greater additive genetic variance from X-linked loci in males than in females (Sidorenko et al., 2019; Jiang et al., 2025).

This is especially relevant for brain-related traits. A 2025 UK Biobank brain XWAS found that many brain imaging traits fit a full-dosage-compensation model, with 25 male-specific non-pseudoautosomal trait-locus pairs vs. only 5 female-specific ones (Jiang et al., 2025). Recent UK Biobank and FinnGen analyses also report a pronounced male bias in X-linked heritability (Fu et al., 2025).

**Important caveats:** The X chromosome contributes only ~3% of autosomal heritability overall. One study found X-linked heritability enrichment for only 2.9% of brain imaging traits, while 39.3% showed depletion. A narrower human study did not find evidence that female X-inactivation averaging explained greater male variance in childhood cognitive scores (Giummo & Johnson, 2012). Cross-species evidence is consistent with a hemizygosity mechanism (variability shifts toward the heterogametic sex in birds), but bird dosage compensation differs from mammalian X-inactivation (Reinhold & Engqvist, 2013).

The strongest defensible claim is that **male hemizygosity together with female X-chromosome inactivation is a plausible, biologically grounded, and probably undermeasured partial contributor** — not a complete explanation.

See the full discussion with references on the [Explanations page](https://smkwray.github.io/sexvary/explanations.html).

</details>

## Limits and interpretation

This project is descriptive. It does not establish biological or causal mechanisms, and observed score variance on public-use instruments does not equal innate capacity. Public-use data impose constraints on survey design handling and harmonization. Some scales are bounded or heaped. The headline pattern — male-greater variability in most cells — is common but not universal. Early reading in ECLS-K is the clearest counterexample, and any interpretation should account for it.

## Reproducibility

The analysis pipeline is written in Python. Results are regenerated from public-use source data via `make backend`. Raw data files are not redistributed; see the [Datasets page](https://smkwray.github.io/sexvary/datasets.html) for acquisition details. For developer setup instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
# Quick start
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .[dev]
make test
make backend      # runs all available pipelines
```

## Project site

The [interactive project site](https://smkwray.github.io/sexvary/) includes:

- Interactive forest plots and bar charts with hover tooltips (ECharts)
- Variance ratio tables with evidence-tier badges
- Age-profile small multiples across all datasets
- Robustness comparison charts
- Full methods and limits documentation

## License

MIT

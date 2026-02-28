# Governance Instruments as Learning Architecture: Evidence from World Bank Program-for-Results

**Dr. Marc Bara** · ESADE Business School · marcoantonio.bara@esade.edu
*Draft submitted to Project Management Journal, February 2026*

---

## What this paper argues

Choosing a lending instrument is not just a financing decision. It is a governance architecture that either makes organizational learning structurally unavoidable or leaves it optional.

The World Bank's **Program-for-Results (PforR)** instrument conditions disbursements on verified results. Projects cannot access funds until independent agents confirm achievement of pre-agreed indicators. This makes measurement, interpretation, and demonstration of progress a *financial* requirement, not a reporting obligation.

This paper tests whether that structural difference translates into better M&E quality and better project outcomes, and whether M&E quality is the mechanism through which the performance premium operates.

**It does, and it is.**

---

## Key results

Using the IEG project ratings dataset (n = 5,757 completed World Bank projects, 1995–2020):

| Model | Dependent variable | PforR β | SE | p |
|---|---|---|---|---|
| 1 | M&E Quality (1–4 scale) | +0.256 | 0.044 | <0.001 |
| 2 | Outcome rating (1–6, total effect) | +0.376 | 0.075 | <0.001 |
| 3 | Outcome rating (1–6, direct effect) | +0.169 | 0.062 | 0.007 |

All models include region fixed effects and HC3 robust standard errors.

> **55.1% of the PforR outcome premium is mediated through M&E quality.**
> The remaining 44.9% reflects direct effects (likely ownership incentives and supervision quality).

---

## Repository structure

```
analysis/
  01_exploration.py    :descriptive statistics, variable distributions, trends
  02_regression.py     :OLS models + product-of-coefficients mediation analysis
data/
  .gitkeep             :folder tracked; data file gitignored (download below)
paper/                 :gitignored:LaTeX source + compiled PDF
```

---

## Reproduce the results

**1. Download the data**

```
https://ieg.worldbankgroup.org/sites/default/files/Data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx
```

Save as `data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx`.

**2. Install dependencies**

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

**3. Run**

```bash
python analysis/01_exploration.py   # descriptive stats
python analysis/02_regression.py    # main results + mediation
```

No other dependencies. All scripts are self-contained and print results to stdout.

---

## How to cite

```
Bara, M. (2026). Governance Instruments as Learning Architecture:
Evidence from World Bank Program-for-Results.
Draft submitted to Project Management Journal.
```

---

## Related literature

The closest published precedent is Heinzel & Liese (2021, *Review of International Organizations*), who find that supervision quality predicts outcomes in IPF but not DPF operations. This paper extends that finding to PforR, uses M&E quality (not staff supervision) as the key variable, and models the mediation mechanism explicitly.

The paper builds on the governance-as-learning-architecture tradition: Brunet (2019, IJPM), Brunet & Choinière (2025, PMJ), Crawford & Helm (2009, PMJ), and the results-based management framework of Ika & Lytvynov (2011, PMJ).

# Cross-Population Evaluation of Explainable AI Frameworks for Heart Disease Prediction

A fully code-driven analysis pipeline: everything below `data/raw/` is generated, in a pinned
environment, and verified to rebuild byte-identically from scratch.

**Team:** Aarushi Batra (230902022) · Ahaan Anand Shah (230902072) · Joshi Arya Vivek (230902064)

**Guide:** Dr. Dashrathraj K Shetty
Minor Project — Data Science minor specialisation, Department of Biomedical Engineering.

---

## What this study asks

Machine learning predicts heart disease accurately but opaquely. Explainable AI (XAI) answers
that, but every prior XAI framework for heart disease was validated on a **single population**,
so nobody knew whether the explanations survive a change of patient cohort.

- **RQ1 — Cross-population consistency.** Do XAI explanations agree across two geographically
  distinct cardiac cohorts (Cleveland, USA vs Hungary)?
- **RQ2 — Framework comparison.** Does adding SHAP produce more consistent explanations than
  LIME+PFI alone?
- **RQ3 — Clinical validation.** Do the features the model relies on match established cardiac
  risk knowledge, and where the cohorts diverge, is that real or a data artefact?

**Framework A** = LIME + PFI  ·  **Framework B** = SHAP + LIME + PFI.

The contribution is not a new XAI method — it is the first cross-population test of these
frameworks and the first head-to-head of A vs B on the same UCI datasets.

---

## The data

### Source

Everything in this study comes from the **UCI Heart Disease dataset**, donated to the UCI Machine
Learning Repository in 1989 and still the dominant benchmark for cardiac classification.

| | |
|---|---|
| Repository page | https://archive.ics.uci.edu/dataset/45/heart+disease |
| DOI | [10.24432/C52P4X](https://doi.org/10.24432/C52P4X) |
| Donated | 1989 (documentation dated 22 July 1988, David Aha) |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Files used here | `processed.cleveland.data`, `processed.hungarian.data` |
| Also present, unused | `processed.switzerland.data`, `processed.va.data` |

The raw files are committed under `data/raw/` and are never modified. Everything else in `data/`
is regenerated from them by `code/phase0.py` and `code/data.py`.

### Attribution required by the data providers

The dataset documentation (`data/raw/heart-disease.names`) carries an explicit **publication
request** from the people who collected the records. Any publication using this data must name the
principal investigator at each contributing institution:

- **Hungarian Institute of Cardiology, Budapest** — Andras Janosi, M.D.
- **University Hospital, Zurich, Switzerland** — William Steinbrunn, M.D.
- **University Hospital, Basel, Switzerland** — Matthias Pfisterer, M.D.
- **V.A. Medical Center, Long Beach and Cleveland Clinic Foundation** — Robert Detrano, M.D., Ph.D.

This is a condition of use, not a courtesy. It must appear in the paper.

### Which files, and why the "processed" ones

Each of the four cohorts exists in two forms: a full file with **76 raw attributes** and a
`processed.*` file carrying the **14 attributes** used in essentially all published work on this
data. We use the `processed.*` files.

That choice matters for one specific reason. Cholesterol recorded as `0`, meaning "not measured"
rather than a real value, is a well-known hazard in this dataset family — a zero that silently
corrupts any imputation or model that trusts it. It is present in the **raw** Hungarian file. The
**processed** files correctly encode those values as `?`. `code/phase0.py` checks for impossible
zeros in `chol`, `trestbps` and `thalach` and finds none in either cohort, so the hazard does not
apply to this analysis.

### Cohorts analysed

| | Cleveland | Hungary |
|---|---|---|
| Source file | `processed.cleveland.data` | `processed.hungarian.data` |
| Rows as delivered | 303 | 294 |
| After de-duplication | 303 | 293 (one exact duplicate removed) |
| Disease prevalence | 45.9% | 36.2% |
| Missing values | none, after the Strategy-A column drop | light; see below |

Switzerland (123 rows) and Long Beach VA (200 rows) are committed for completeness because they
are part of the same donation and are the natural extension to a third and fourth cohort. **No
code path reads them.**

### Attributes

Of the 14 attributes in the processed files, this study uses **10 features plus the target**.
Codings are those defined in `heart-disease.names`:

| Feature | Meaning | Coding |
|---|---|---|
| `age` | Age | years |
| `sex` | Sex | 1 = male, 0 = female |
| `cp` | Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| `trestbps` | Resting blood pressure on admission | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | Resting electrocardiographic result | 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy (Estes' criteria) |
| `thalach` | Maximum heart rate achieved | bpm |
| `exang` | Exercise-induced angina | 1 = yes, 0 = no |
| `oldpeak` | ST depression induced by exercise relative to rest | mm |
| `num` (target) | Angiographic disease status | 0 = < 50% diameter narrowing; 1–4 = > 50% narrowing. **Binarised as `num > 0`.** |

**Three attributes are deliberately dropped**, and this is the single most consequential decision
in the study:

| Dropped | Meaning | Missing in Cleveland | Missing in Hungary |
|---|---|---|---|
| `slope` | Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) | 0.0% | **64.6%** |
| `ca` | Number of major vessels (0–3) coloured by fluoroscopy | 1.3% | **99.0%** |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) | 0.7% | **90.5%** |

In Hungary, `ca` is missing for 291 of 294 patients — the column effectively does not exist there.
Imputing a 99%-absent column would not recover the feature, it would manufacture one, and any
importance later attributed to it would measure the imputation rather than the patients. Since the
question is whether explanations agree *across populations*, the comparison has to run on a feature
set both populations actually possess.

The cost is stated openly: `ca` and `thal` are among the strongest predictors reported in
Cleveland-only work. **Every finding in this project is scoped to the shared ten-feature space.**

### Residual missingness

After the column drop, Cleveland has no missing values at all — those three columns held its only
gaps. Hungary retains light missingness in 33 of 294 records (11.2%), imputed with statistics fit
**separately within each cohort** so population-specific distributions are preserved:

| Feature | % missing in Hungary | Imputed with | Value used |
|---|---|---|---|
| `chol` | 7.8% | median | 243 |
| `fbs` | 2.7% | mode | 0 |
| `trestbps` | 0.3% | median | 130 |
| `restecg` | 0.3% | mode | 0 |
| `thalach` | 0.3% | median | 140 |
| `exang` | 0.3% | mode | 0 |

Phase 6 revisits `chol` directly, since it is the only feature with meaningful imputation exposure
and therefore the only one where a real-versus-artefact question genuinely arises.

### Integrity

The two files the analysis reads, exactly as committed:

```
2d91a8ff69cfd9616aa47b59d6f843db  data/raw/processed.cleveland.data   303 rows
22e96bee155b5973568101c93b3705f6  data/raw/processed.hungarian.data   294 rows
```

Verify with `md5sum data/raw/processed.*.data`. If these differ from the values above, the inputs
have changed and no downstream result should be trusted until that is explained.

### How to cite

> A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning
> Repository, 1989, doi: 10.24432/C52P4X.

The clinical study behind the data collection is worth citing alongside it, since it was itself an
international multi-site validation:

> R. Detrano et al., "International application of a new probability algorithm for the diagnosis of
> coronary artery disease," *American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989.

### Ethics and provenance

The records are fully de-identified, were collected by the four institutions named above, and are
publicly redistributable under CC BY 4.0. This study is a secondary analysis of public data; it
involved no human participants, no new data collection and no attempt at re-identification.

---

## Quick start

```bash
# 1. environment (Python 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. full pipeline — run from THIS directory, in this order
.venv/bin/python code/phase0.py              # raw -> interim, missingness audit
.venv/bin/python code/data.py                # interim -> clean  (Phase 1)
.venv/bin/python code/models.py              # 7 classifiers, 10-fold CV, model selection
.venv/bin/python code/explain.py             # Framework A: PFI + LIME + stability protocol
.venv/bin/python code/frameworkB.py          # Framework B: + SHAP        [needs Phase 3 tables]
.venv/bin/python code/phase5.py compute 0 30 # 30-split robust analysis   (~6 min, chunkable)
.venv/bin/python code/phase5.py aggregate    # summary + Wilcoxon + 6 figures
.venv/bin/python code/phase6.py              # clinical validation
```

Every script uses paths relative to the repository root, so **run them from this directory**
(`python code/x.py`, never `cd code && python x.py`). Python puts `code/` on `sys.path`
automatically, which is how the modules find each other.

All results are deterministic under the fixed seeds (seed 42 throughout; Phase 5 uses split
seeds 0–29). Re-running the pipeline reproduces every table and figure exactly.

### Re-running Phase 5 cleanly

`phase5.py compute` **appends** to disk, and is chunkable/resumable by design
(`compute 0 10`, `compute 10 20`, `compute 20 30`). Before a clean re-run, delete
`results/tables/phase5_raw_per_split.csv` and `phase5_importances_long.csv`.
`aggregate` de-duplicates on `split`, so an accidental double-run self-corrects — but a stale
file written under different settings would not be caught. Confirm it prints
`aggregating over 30 splits`.

---

## Layout

```
.
├── README.md
├── reproduction-notes.md     full rebuild guide: setup, run order, checkpoints, troubleshooting
├── requirements.txt          pinned; install from this, not by hand
├── code/
│   ├── phase0.py             Phase 0 — raw -> interim + missingness audit
│   ├── data.py               Phase 1 — interim -> clean (Strategy A + imputation)
│   ├── models.py             Phase 2 — 7 classifiers, 10-fold CV, model selection
│   ├── consistency.py        shared metrics (Spearman, Kendall, top-k overlap, Jaccard)
│   ├── explain.py            Phase 3 — Framework A (PFI, LIME, LIME stability protocol)
│   ├── frameworkB.py         Phase 4 — adds SHAP -> Framework B
│   ├── phase5.py             Phase 5 — 30-split robust analysis (resumable CLI)
│   └── phase6.py             Phase 6 — clinical validation
├── data/
│   ├── raw/                  UCI donation [PRISTINE — never edit, never generated]
│   ├── interim/              14 cols, '?'->NaN, nothing dropped     [generated by phase0.py]
│   └── clean/                10 features + num, UNSCALED            [generated by data.py]
├── results/
    ├── tables/               all CSVs      [generated]
    └── figures/              all PNGs      [generated]
```

Only `data/raw/` and `code/` are inputs. **Everything else is generated** — deleting
`data/interim/`, `data/clean/` and `results/` and re-running the pipeline rebuilds them all.

### Dependency chain — respect this order

```
phase0.py → data.py → models.py → explain.py → frameworkB.py → phase5.py → phase6.py
                                               (reads Phase-3   (imports from   (reads phase5_
                                                CSV tables)      explain.py +    mean_importances
                                                                 frameworkB.py)   .csv)
```

`frameworkB.py` reads `frameworkA_pfi.csv` and `frameworkA_lime_importance.csv` from disk, so
`explain.py` must run first. `phase5.py` imports `load_split`, `fit_rf`, `run_pfi`, `_explainer`
from `explain.py` and `shap_explain` from `frameworkB.py`. `phase6.py` depends on
`phase5_mean_importances.csv`.

---

## Invariants — do not break these

Each was a deliberate decision. Several look like tidying opportunities and are not.

1. **`data/raw/` is immutable.** All transformation flows raw → interim → clean via code.
2. **`data/clean/*.csv` are UNSCALED and must stay that way.** `StandardScaler` is fit *inside*
   CV folds (`models.py` wraps scale-sensitive models in a `Pipeline`). Baking a scaler into the
   clean files reintroduces train/test leakage and is model-inappropriate for trees. This looks
   like a missing preprocessing step. It is not.
3. **`shuffle=True` in every `StratifiedKFold`.** The Hungarian source file is *sorted by age*;
   unshuffled folds are age-biased.
4. **Identical feature order across cohorts**, defined once as `FEATURES` in `data.py`.
   Importance vectors are compared element-wise across cohorts — reordering silently corrupts
   every consistency number.
5. **Identical preprocessing for both cohorts.** Imputation *statistics* are fit per cohort
   (population-specific), but the *rule* is the same.
6. **Seed 42 everywhere**; Phase 5 uses split seeds 0–29.
7. **The 10-feature set is fixed:** `age sex cp trestbps chol fbs restecg thalach exang oldpeak`.
   `slope`, `ca`, `thal` are dropped by design (Strategy A).
8. **Explanations are computed on held-out test data**, never on training data.

---

## Key design decisions

**Strategy A — drop `slope`, `ca`, `thal`.** In Hungary these are 64.6% / 99.0% / 90.5% missing;
in Cleveland they are essentially complete. Imputing a 99%-missing column fabricates the feature,
so any importance it showed would be an artefact of the imputation rather than of the patients.
A truthful cross-population comparison requires a feature set both cohorts actually possess.
Cost, stated openly: `ca` and `thal` are two of the strongest predictors in prior Cleveland-only
work, so all findings concern the shared 10-feature space.

**Random Forest as the explained model.** The seven classifiers are statistically tied on AUC
(top four within 0.007, against a fold std of ~0.05). RF was chosen over top-ranked LR because a
linear model would hollow out the study: LIME fits a *linear local surrogate*, so on an already
linear model LIME/SHAP/PFI would trivially agree and the A-vs-B comparison would have nothing to
discriminate. RF is also the single best model on Hungary, and unlocks SHAP's exact, deterministic
`TreeExplainer`.

**"Consistency" is operationalised** as Spearman and Kendall rank correlation plus top-k overlap
and Jaccard at k = 3, 5, between two importance vectors. Framework-level consistency uses a
consensus rank (mean feature rank across a framework's methods).

> ⚠️ **Terminology trap.** SHAP's *consistency axiom* is a game-theoretic property about how
> attributions behave when the model changes. **Cross-population consistency** is this project's
> own measure of whether explanations agree across cohorts. Always write "cross-population
> consistency" for the latter.

---

## Environment notes

Python 3.12, with versions pinned in `requirements.txt`. SHAP and LIME both have breaking changes
between releases, and a version mismatch between team members silently changes figures — install
from the pinned file rather than by hand.

- SHAP 0.52 `TreeExplainer` returns `.values` shaped `(n_samples, n_features, n_classes)`.
  **Always extract class index 1 (disease).** Older versions return a list of arrays — check the
  shape before indexing if the version changes.
- If `pip install` is refused with a PEP 668 "externally-managed-environment" error, use a venv
  (as above) or pass `--break-system-packages`.
- `phase5.py` is chunked and resumable because detached jobs were unreliable in the original
  build environment. Keep that — it is robust anywhere.

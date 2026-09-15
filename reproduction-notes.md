# Reproduction Notes

Everything needed to rebuild this project's findings from scratch and confirm you got the same
answers. Written for someone who has the repository and nothing else.

The short version: install from the pinned requirements, run eight commands in order, and compare
your output against the checkpoints below. The whole pipeline takes about **6 minutes**.

---

## 1. What you need

- **Python 3.12.** The pin matters (see §7); 3.12.14 is the version of record.
- **~500 MB** of disk for the virtual environment.
- No GPU, no network access after install, no credentials, no external data downloads. The raw
  UCI files are committed in `data/raw/`.

Nothing else is required. The analysis is small — two cohorts of about 300 patients each.

---

## 2. Setup

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

If you use `pip` instead of `uv`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Install from `requirements.txt`, not by hand.** SHAP and LIME both introduce breaking changes
between releases, and a version mismatch does not raise an error — it silently changes the numbers.
That is the entire reason the file is pinned.

Confirm the environment:

```bash
.venv/bin/python -c "import sklearn, shap, lime, xgboost; print(sklearn.__version__, shap.__version__, xgboost.__version__)"
# expect: 1.9.1 0.52.0 3.3.0
```

---

## 3. Running the pipeline

Run every command **from the repository root**, as `python code/x.py` — never `cd code && python x.py`.
Python puts `code/` on `sys.path` based on the script's own location, which is how the modules find
each other, while the relative data paths (`data/clean/...`) resolve against your working directory.
Running from inside `code/` breaks the second half of that.

```bash
.venv/bin/python code/phase0.py              # raw → interim, missingness audit      ~2 s
.venv/bin/python code/data.py                # interim → clean                       ~1 s
.venv/bin/python code/models.py              # 7 classifiers, 10-fold CV, selection  ~30 s
.venv/bin/python code/explain.py             # Framework A: PFI + LIME + stability   ~65 s
.venv/bin/python code/frameworkB.py          # Framework B: + SHAP                   ~7 s
.venv/bin/python code/phase5.py compute 0 30 # 30-split robust analysis              ~5 min
.venv/bin/python code/phase5.py aggregate    # summary, Wilcoxon tests, 6 figures    ~3 s
.venv/bin/python code/phase6.py              # clinical validation                   ~2 s
```

The order is a real dependency chain, not a convention:

```
phase0 → data → models → explain → frameworkB → phase5 → phase6
                                   (reads       (imports from   (reads phase5_
                                    Phase-3      explain.py +    mean_importances
                                    CSVs)        frameworkB.py)  .csv)
```

`frameworkB.py` loads `frameworkA_pfi.csv` and `frameworkA_lime_importance.csv` **from disk**, so
`explain.py` must have run first. `phase6.py` consumes `phase5_mean_importances.csv`, which only
exists after `phase5.py aggregate`.

Phase 5 is the long step and is deliberately **chunked and resumable** — you can split it across
sessions:

```bash
.venv/bin/python code/phase5.py compute 0 10
.venv/bin/python code/phase5.py compute 10 20
.venv/bin/python code/phase5.py compute 20 30
```

---

## 4. Checkpoints — what you should see

Compare against these as you go. If a checkpoint matches, everything upstream of it is correct.

### Phase 0
```
Cleveland   303 rows x 14 columns | exact duplicate rows: 0
Hungary     294 rows x 14 columns | exact duplicate rows: 1
impossible zeros: none in either cohort
>50% missing in at least one cohort -> DROP: ['slope', 'ca', 'thal']
```
Hungary missingness: `slope` 64.6%, `ca` 99.0%, `thal` 90.5%, `chol` 7.8%, `fbs` 2.7%.

### Phase 1
```
[Cleveland] 303 -> 303 rows (dropped 0 duplicate), 45.9% positive   — no imputation needed
[Hungary]   294 -> 293 rows (dropped 1 duplicate), 36.2% positive
            trestbps <- 130, chol <- 243, thalach <- 140, fbs <- 0, restecg <- 0, exang <- 0
```

### Phase 2
Mean AUC across both cohorts:
```
LR 0.878 · RF 0.874 · NB 0.874 · SVM 0.871 · KNN 0.863 · XGBoost 0.849 · DT 0.698
>>> Selected architecture: RF (AUC_avg=0.874, F1_avg=0.727, best on Hungary at 0.891)
```
The script prints LR as the highest-AUC model **and** RF as the selected one. That is correct and
intentional — see `reports/docs/phase2-model-selection.md` for the tie-break rule.

### Phase 3
```
[Cleveland] test AUC=0.947 | LIME rank-stability=0.983 top3-Jaccard=0.904
[Hungary]   test AUC=0.879 | LIME rank-stability=0.985 top3-Jaccard=1.0
PFI cross-population Spearman 0.552 · LIME 0.612 · PFI↔LIME 0.733 in both cohorts
```

### Phase 4
```
SHAP cross-population Spearman 0.842
Cleveland: PFI↔SHAP 0.842, LIME↔SHAP 0.903
Hungary:   PFI↔SHAP 0.430, LIME↔SHAP 0.733
Framework A consensus 0.602  vs  Framework B consensus 0.571
```
Framework B looking *worse* here is the expected single-split result, and Phase 5 reverses it. The
Hungary `PFI↔SHAP 0.430` is likewise expected and resolves to 0.746 over 30 splits.

### Phase 5 — the headline
`aggregate` must print `aggregating over 30 splits`. If it says any other number, see §6.

```
PFI   Spearman 0.624 ± 0.054    top3 2.00/3   top5 3.73/5
LIME  Spearman 0.598 ± 0.054    top3 2.07/3   top5 3.93/5
SHAP  Spearman 0.838 ± 0.025    top3 2.13/3   top5 4.27/5
A     Spearman 0.638 ± 0.039    top3 2.00/3   top5 3.93/5
B     Spearman 0.749 ± 0.028    top3 2.10/3   top5 4.43/5

Paired Wilcoxon, all p < 0.001:
  SHAP vs PFI  +0.214 | SHAP vs LIME  +0.241 | Framework B vs A  +0.111
```

### Phase 6
```
Cleveland top-5 (consensus): ['cp', 'oldpeak', 'sex', 'thalach', 'exang']  -> 5/5 established
Hungary   top-5 (consensus): ['cp', 'oldpeak', 'exang', 'sex', 'thalach']  -> 5/5 established
Cleveland: age 54.4±9.0 [29-77] | thalach 150±23 | positives 46%
Hungary:   age 47.8±7.8 [28-66] | thalach 139±24 | positives 36%
```

---

## 5. Verifying determinism yourself

> **The claim:** delete every generated file, rebuild from `data/raw/` and `code/` alone, and every
> table and figure returns **byte-identical**.

To test it:

```bash
git status --porcelain          # confirm clean first
rm -rf data/interim data/clean results
# ...re-run all eight commands from §3...
git status --porcelain          # expect: empty
```

An empty `git status` means all 22 tables and 22 figures regenerated bit-for-bit. This has been
verified, including across a move of the project directory, which rules out any hidden dependence
on absolute paths.

Worth knowing: **LIME reproduces exactly**, despite being stochastic by construction. The 180
explanations behind the Phase-3 stability protocol and all 30 Phase-5 splits return identical
values. LIME's inherent variability is real and is measured deliberately in Phase 3 — but that is
variability under re-sampling, not irreproducibility.

### Where the claim stops

Determinism is verified **on one machine under the pinned versions**. It has not been tested across
operating systems or CPU architectures, where floating-point reduction order in NumPy/BLAS can
differ with thread counts or hardware and move low-order digits. If you reproduce this on different
hardware and see differences in the third or fourth decimal, that is the likely cause and is not a
failure of the analysis. Differences in the *first* decimal, or a changed conclusion, mean something
is genuinely wrong — check your library versions first.

### Every source of randomness, and how it is controlled

| Source | Control |
|---|---|
| `train_test_split` (80/20 explanation split) | `random_state=42`; Phase 5 sweeps seeds 0–29 explicitly |
| `StratifiedKFold` (Phase 2) | `shuffle=True, random_state=42` — shuffling is **required**, see below |
| `RandomForestClassifier` | `random_state=42` |
| `permutation_importance` (PFI) | `random_state=42`, 30 repeats |
| `LimeTabularExplainer` | `random_state=42` at construction |
| SHAP `TreeExplainer` | Exact and deterministic by design — no sampling, no seed needed |
| `shap.plots.beeswarm` | Not seedable; handled explicitly, see §7.1 |

**Fold shuffling is not optional.** The Hungarian source file is sorted by age. Without
`shuffle=True`, every cross-validation fold would be an age band and every Phase-2 estimate would be
biased by construction.

---

## 6. Troubleshooting

**`aggregate` prints a number other than 30.**
`phase5.py compute` *appends* to disk — that is what makes it resumable. You have either a partial
run or leftover rows. For a clean rebuild, delete both files and re-run `compute 0 30`:
```bash
rm results/tables/phase5_raw_per_split.csv results/tables/phase5_importances_long.csv
```
`aggregate` de-duplicates on `split`, so an accidental double-run self-corrects — but a stale file
written under *different settings* will not be caught. Always check the printed count.

**`FileNotFoundError` on `frameworkA_pfi.csv`.**
You ran `frameworkB.py` before `explain.py`. Follow the order in §3.

**`ModuleNotFoundError` for a project module (e.g. `No module named 'explain'`).**
You ran from inside `code/`. Run from the repository root as `python code/x.py`.

**An `AssertionError` from `phase0.py`, `data.py` or `models.py`.**
These are deliberate guards, not bugs — see §8. Read the message; it names the assumption that
broke.

**Figures look different but CSVs match.**
Check your matplotlib version. If only the two SHAP beeswarms differ, see §7.1.

---

## 7. Library traps

### 7.1 `shap.plots.beeswarm` is not reproducible by default
It shuffles and jitters its points using NumPy's **global** random state, and the modern API exposes
no `rng` argument — only the legacy `summary_legacy()` does. Left alone, the two beeswarm PNGs are
the only artefacts in the project that change between otherwise identical runs. The SHAP values
behind them are unaffected; only the pixel layout moves.

`frameworkB.py` therefore calls `np.random.seed(SEED)` immediately before plotting, with the reason
recorded inline. **Do not remove that line.**

### 7.2 SHAP's output shape is version-dependent
SHAP 0.52 `TreeExplainer` returns `.values` shaped `(n_samples, n_features, n_classes)`, and the code
extracts **class index 1 (disease)**. Older versions return a *list* of per-class arrays. If the pin
changes, check the shape before indexing — a wrong assumption here does not raise, it silently
explains the wrong class.

### 7.3 `SVC(probability=True)` is deprecated
scikit-learn 1.9 deprecated it; removal is scheduled for 1.11, with
`CalibratedClassifierCV(SVC(), ensemble=False)` as the replacement. It affects only the SVM row in
Phase 2 and changes nothing at the pinned version. **Do not upgrade scikit-learn** without re-running
and re-checking Phase 2.

### 7.4 matplotlib removed `boxplot(labels=)`
Renamed to `tick_labels=`, removed outright in 3.11. The three boxplot figures use the current
spelling, so matplotlib older than 3.9 will reject them.

---

## 8. Self-checking behaviour

Three assertions guard assumptions the whole study rests on. They fail loudly rather than degrading
quietly:

- **`phase0.py`** — asserts the >50%-missing rule still selects exactly `slope`, `ca`, `thal`. If the
  input data changed, Strategy A would silently become a different strategy.
- **`data.py`** — asserts zero residual NaNs after imputation, per cohort.
- **`models.py`** — asserts the documented selection rule still returns Random Forest, the model the
  entire XAI pipeline hardcodes. Without it, Phase 2 could drift out of sync with Phases 3–6 and
  nothing would say so.

---

## 9. What is input and what is generated

| Path | Status |
|---|---|
| `data/raw/` | The pristine UCI donation — 4 `processed.*.data` files plus `heart-disease.names` and `WARNING`. **Never edited.** |
| `code/` | The eight analysis modules. |
| everything else | **Generated.** Safe to delete; the pipeline rebuilds it. |

`data/interim/`, `data/clean/`, `results/tables/` and `results/figures/` are all outputs. Nothing in
this project is hand-edited or produced by a step that is not in `code/`.

The analysis uses **Cleveland and Hungary only**. The Switzerland and Long Beach VA files sit in
`data/raw/` because they are part of the same UCI donation and are the natural extension to a third
and fourth cohort, but no code path reads them.

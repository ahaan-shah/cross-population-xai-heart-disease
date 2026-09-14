"""
data.py  —  Phase 1 preprocessing for the Cross-Population XAI heart-disease project.

Takes the INTERIM labeled datasets (14 columns, '?' already converted to NaN) and
produces the CLEAN, modeling-ready datasets under **Strategy A**.

Pipeline (applied identically to both cohorts; imputation is fit PER cohort):
    1. Drop exact duplicate rows.
    2. Binarize target:  num > 0  ->  1     (Cleveland is 0-4; Hungary already 0/1).
    3. Strategy A: drop the heavily-missing columns  [slope, ca, thal].
    4. Light imputation of residual missingness, fit on each cohort separately:
           continuous  [trestbps, chol, thalach] -> median
           categorical [fbs, restecg, exang]     -> most-frequent (mode)
       (age, sex, cp, oldpeak have no residual missingness in either cohort.)
    5. Enforce identical column order across both cohorts.
    6. Tidy dtypes: ordinal/binary/target -> int, continuous measurements -> float.

NOT done here (deliberately):
    * Feature scaling. StandardScaler is fit INSIDE cross-validation folds in Phase 2,
      to avoid train/test leakage, and because only some models (SVM/KNN/LR) need it.
      Baking a global scaler into the saved file would leak and would be model-agnostic
      in the wrong way. See the methodology note for the leakage rationale.

Run as a script:   python data.py
Or import:         from data import preprocess, run
"""

from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer

# Column order in the raw/interim files (14 columns, no header originally)
RAW_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

DROP_COLS   = ['slope', 'ca', 'thal']                                  # Strategy A
FEATURES    = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
               'restecg', 'thalach', 'exang', 'oldpeak']               # 10 features, fixed order
CONT_IMPUTE = ['trestbps', 'chol', 'thalach']                          # median-imputed
CAT_IMPUTE  = ['fbs', 'restecg', 'exang']                              # mode-imputed
INT_COLS    = ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'num']   # cast to int at the end
# (trestbps, chol, thalach, oldpeak remain float — they are continuous measurements)


def preprocess(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Transform one interim cohort DataFrame into its clean, modeling-ready form."""
    df = df.copy()

    # 1. exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    # 2. binarize the target
    df['num'] = (df['num'] > 0).astype(int)

    # 3. Strategy A: drop heavily-missing columns
    df = df.drop(columns=DROP_COLS)

    # 4. light imputation — fit on THIS cohort only (per-population statistics)
    imputed = {}
    cont_missing = [c for c in CONT_IMPUTE if df[c].isna().any()]
    cat_missing  = [c for c in CAT_IMPUTE  if df[c].isna().any()]
    if cont_missing:
        imp = SimpleImputer(strategy='median')
        df[cont_missing] = imp.fit_transform(df[cont_missing])
        imputed.update({c: ('median', float(v)) for c, v in zip(cont_missing, imp.statistics_)})
    if cat_missing:
        imp = SimpleImputer(strategy='most_frequent')
        df[cat_missing] = imp.fit_transform(df[cat_missing])
        imputed.update({c: ('mode', float(v)) for c, v in zip(cat_missing, imp.statistics_)})

    # 5 & 6. fixed feature order + tidy dtypes
    df = df[FEATURES + ['num']]
    for c in INT_COLS:
        df[c] = df[c].round().astype(int)

    assert df.isna().sum().sum() == 0, f"[{cohort}] residual NaNs remain after imputation"

    pos = df['num'].mean() * 100
    print(f"[{cohort}] {n_before} -> {len(df)} rows "
          f"(dropped {n_dupes} duplicate), {pos:.1f}% positive")
    if imputed:
        for col, (how, val) in imputed.items():
            print(f"         imputed {col:8s} <- {how} = {val:g}")
    else:
        print("         no imputation needed")
    return df


def run(interim_dir='data/interim', clean_dir='data/clean'):
    """Read interim labeled CSVs, preprocess, and write clean CSVs."""
    interim_dir, clean_dir = Path(interim_dir), Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ('Cleveland', 'cleveland_labeled.csv', 'cleveland_clean.csv'),
        ('Hungary',   'hungarian_labeled.csv', 'hungary_clean.csv'),
    ]
    for cohort, infile, outfile in jobs:
        df = pd.read_csv(interim_dir / infile)   # interim already has headers + NaN for '?'
        clean = preprocess(df, cohort)
        clean.to_csv(clean_dir / outfile, index=False)
        print(f"         saved -> {clean_dir / outfile}\n")


if __name__ == '__main__':
    run()

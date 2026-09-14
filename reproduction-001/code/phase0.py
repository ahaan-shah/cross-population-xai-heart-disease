"""
phase0.py  —  Phase 0: raw -> interim, plus the data audit that decides Strategy A.

This is the step that was missing from the original build: the raw -> interim
transformation and the missingness audit were performed ad hoc and never saved as
code, so `data/interim/` and `results/tables/missingness_audit.csv` could not be
regenerated. This module makes the whole chain reproducible from `data/raw/` alone.

Loading convention (fixed by the UCI donation format):
    * the processed.*.data files have NO header row
    * 14 columns, named manually in the canonical UCI order (RAW_COLS in data.py)
    * missing values are the literal string '?'  ->  NaN

What it does
------------
1. Loads both raw cohort files and writes `data/interim/*_labeled.csv`
   (14 columns, '?' -> NaN, NOTHING dropped -- a durable, auditable checkpoint).
2. Audits, and prints, the five things the Phase-0 note reports:
     - dataset shape and exact-duplicate count
     - % missing per feature per cohort      -> the table that decides Strategy A
     - disguised missingness (physiologically impossible zeros)
     - target distribution, raw and binarised
     - value ranges, for cross-cohort comparability
   plus residual missingness after the Strategy-A column drop.
3. Writes `results/tables/missingness_audit.csv` and `missingness_audit.png`.

IMPORTANT — the audit is computed on the interim data, BEFORE de-duplication,
i.e. on 303 Cleveland / 294 Hungary rows. De-duplication belongs to Phase 1.
This is what the published percentages correspond to (e.g. Hungary ca = 291/294
= 99.0%); auditing post-dedup would shift every Hungarian figure slightly.

Run:  python code/phase0.py        (from the repository root)
"""

from pathlib import Path
import pandas as pd

from data import RAW_COLS, DROP_COLS, FEATURES

# Features carried through the audit = everything except the target.
AUDIT_COLS = [c for c in RAW_COLS if c != 'num']

# Columns where a 0 would be physiologically impossible and therefore indicates
# "not recorded" rather than a real measurement. This is the disguised-missingness
# check: it is a genuine problem in the raw 76-attribute files, and the point of
# the check is to confirm the *processed* files do not have it.
ZERO_IMPOSSIBLE = ['trestbps', 'chol', 'thalach']

COHORTS = [
    ('Cleveland', 'processed.cleveland.data', 'cleveland_labeled.csv'),
    ('Hungary',   'processed.hungarian.data', 'hungarian_labeled.csv'),
]


def load_raw(path) -> pd.DataFrame:
    """Read one raw UCI processed file into a 14-column frame with NaN for '?'."""
    return pd.read_csv(path, header=None, names=RAW_COLS, na_values='?')


def audit_cohort(df: pd.DataFrame, cohort: str) -> pd.Series:
    """Print the per-cohort audit and return its % missing per feature."""
    n = len(df)
    n_dupes = int(df.duplicated().sum())
    pct_missing = (df[AUDIT_COLS].isna().mean() * 100).round(1)

    print(f"\n--- {cohort} ---")
    print(f"  shape: {n} rows x {df.shape[1]} columns | exact duplicate rows: {n_dupes}")

    # disguised missingness
    zeros = {c: int((df[c] == 0).sum()) for c in ZERO_IMPOSSIBLE}
    bad = {c: v for c, v in zeros.items() if v}
    print(f"  impossible zeros {ZERO_IMPOSSIBLE}: "
          + (f"FOUND {bad}" if bad else "none (processed file encodes these as '?')"))

    # target distribution, before and after binarisation
    raw_counts = df['num'].value_counts().sort_index().to_dict()
    binary = (df['num'] > 0).astype(int)
    print(f"  target raw {raw_counts}")
    print(f"  target binarised (num>0): {(binary == 0).sum()} no-disease / "
          f"{(binary == 1).sum()} disease ({binary.mean() * 100:.1f}% positive)")

    # residual missingness once Strategy A drops slope/ca/thal
    kept = df[FEATURES]
    rows_hit = int(kept.isna().any(axis=1).sum())
    print(f"  after dropping {DROP_COLS}: {rows_hit}/{n} rows "
          f"({rows_hit / n * 100:.1f}%) still carry missing values "
          f"-> {n - rows_hit} rows would remain if dropped instead of imputed")

    return pct_missing


def value_ranges(frames: dict) -> pd.DataFrame:
    """min-max per kept feature per cohort — the comparability precondition."""
    out = {}
    for cohort, df in frames.items():
        out[cohort] = [f"{df[f].min():g}-{df[f].max():g}" for f in FEATURES]
    return pd.DataFrame(out, index=FEATURES)


def run(raw_dir='data/raw', interim_dir='data/interim',
        out_tables='results/tables', out_figs='results/figures'):
    raw_dir, interim_dir = Path(raw_dir), Path(interim_dir)
    out_tables, out_figs = Path(out_tables), Path(out_figs)
    for d in (interim_dir, out_tables, out_figs):
        d.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 0 — DATA AUDIT ===")

    frames, missing = {}, {}
    for cohort, infile, outfile in COHORTS:
        df = load_raw(raw_dir / infile)
        df.to_csv(interim_dir / outfile, index=False)
        frames[cohort] = df
        missing[cohort] = audit_cohort(df, cohort)
        print(f"  saved -> {interim_dir / outfile}")

    # ---- the deciding table ----
    audit = pd.DataFrame({f'{c}_%miss': s for c, s in missing.items()})
    audit['Worst_of_two'] = audit.max(axis=1)
    audit.to_csv(out_tables / 'missingness_audit.csv')

    print("\n=== MISSINGNESS AUDIT (% missing per feature) ===")
    print(audit.to_string())

    drops = audit.index[audit['Worst_of_two'] > 50].tolist()
    print(f"\n  >50% missing in at least one cohort -> DROP: {drops}")
    assert drops == DROP_COLS, (
        f"Strategy A expects to drop {DROP_COLS}, but the audit indicates {drops}")

    print("\n=== VALUE RANGES (kept features) ===")
    print(value_ranges(frames).to_string())

    _figure(audit, out_figs / 'missingness_audit.png')
    return audit


def _figure(audit: pd.DataFrame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    t = audit.sort_values('Worst_of_two')
    y = np.arange(len(t))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + h / 2, t['Cleveland_%miss'], h, label='Cleveland', color='#4C72B0')
    ax.barh(y - h / 2, t['Hungary_%miss'],  h, label='Hungary',  color='#C44E52')
    ax.axvline(50, ls='--', lw=1, color='grey')
    ax.text(51, len(t) - 1.2, 'Strategy A\ndrop threshold', fontsize=9, color='grey')
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlabel('% missing')
    ax.set_title('Phase 0 — missingness by feature and cohort')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[saved] {path}")


if __name__ == '__main__':
    run()

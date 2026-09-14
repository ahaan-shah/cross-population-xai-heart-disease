"""
phase6.py  —  Phase 6: clinical validation.

Checks whether the features the models rely on match established cardiac risk
knowledge, per cohort, and — critically — separates GENUINE population differences
from DATA ARTEFACTS using two data-driven diagnostics:

  1. Univariate separability: how well does each feature ALONE separate disease
     (single-feature AUC) in each cohort? A feature important in one cohort but
     not the other is a REAL signal difference if its univariate AUC tracks that.
  2. Imputation exposure: how much of each feature was imputed in Hungary? High
     imputation + apparent importance = interpret with caution (possible artefact).

Uses the robust Phase-5 mean importances (averaged over 30 splits).
Run:  python phase6.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak']
METHODS = ['PFI', 'LIME', 'SHAP']
TABLES = Path('results/tables')
FIGS = Path('results/figures')

# established cardiac role of each retained feature (for the report mapping)
CLINICAL = {
    'cp':       ('Chest pain type', 'strong',   'Core symptomatic predictor of coronary disease'),
    'oldpeak':  ('ST depression (exercise)', 'strong', 'Classic ECG marker of exercise-induced ischaemia'),
    'thalach':  ('Max heart rate achieved', 'strong', 'Low peak HR = reduced exercise capacity / chronotropic incompetence'),
    'exang':    ('Exercise-induced angina', 'strong', 'Direct symptom of inducible ischaemia'),
    'age':      ('Age', 'moderate', 'Non-modifiable risk; rises with age'),
    'sex':      ('Sex', 'moderate', 'Male sex confers higher risk'),
    'trestbps': ('Resting blood pressure', 'moderate', 'Hypertension is an established risk factor'),
    'chol':     ('Serum cholesterol', 'moderate', 'Hypercholesterolaemia is a classic risk factor (often weak in UCI)'),
    'restecg':  ('Resting ECG', 'weak', 'Abnormalities indicate underlying disease'),
    'fbs':      ('Fasting blood sugar >120', 'weak', 'Diabetes proxy; typically weak in UCI'),
}


def consensus_rank(mean_imp, cohort):
    ranks = [mean_imp[f'{m}_{cohort}'].rank(ascending=False) for m in METHODS]
    return (sum(ranks) / len(ranks)).round(2)


def uni_auc(df, f):
    """Single-feature separability of disease (direction-agnostic)."""
    a = roc_auc_score(df['num'].values, df[f].values)
    return round(max(a, 1 - a), 3)


def run():
    mean_imp = pd.read_csv(TABLES / 'phase5_mean_importances.csv', index_col=0)
    clev = pd.read_csv('data/clean/cleveland_clean.csv')
    hung = pd.read_csv('data/clean/hungary_clean.csv')
    hlab = pd.read_csv('data/interim/hungarian_labeled.csv')   # pre-imputation, for missingness

    cr = consensus_rank(mean_imp, 'Cleveland')
    hr = consensus_rank(mean_imp, 'Hungary')
    imp_frac = (hlab[FEATURES].isna().mean() * 100).round(1)   # % imputed in Hungary

    rows = []
    for f in FEATURES:
        role, strength, note = CLINICAL[f]
        rows.append({
            'feature': f, 'clinical_role': role, 'established': strength,
            'clev_rank': cr[f], 'hung_rank': hr[f],
            'clev_uniAUC': uni_auc(clev, f), 'hung_uniAUC': uni_auc(hung, f),
            'hung_imputed_%': imp_frac[f],
        })
    tab = pd.DataFrame(rows).sort_values('clev_rank')
    tab.to_csv(TABLES / 'phase6_clinical_validation.csv', index=False)

    # clinical alignment score: fraction of top-5 consensus features that are established (strong/moderate)
    def alignment(rank):
        top5 = rank.sort_values().index[:5]
        est = sum(CLINICAL[f][1] in ('strong', 'moderate') for f in top5)
        return top5.tolist(), est
    c_top, c_est = alignment(cr)
    h_top, h_est = alignment(hr)

    print("=== CLINICAL VALIDATION TABLE ===")
    print(tab.to_string(index=False))
    print(f"\nCleveland top-5 (consensus): {c_top}  -> {c_est}/5 established risk factors")
    print(f"Hungary   top-5 (consensus): {h_top}  -> {h_est}/5 established risk factors")

    # divergence diagnostics for the features that differ most between cohorts
    tab['rank_gap'] = (tab['clev_rank'] - tab['hung_rank']).abs()
    print("\n=== DIVERGENCE DIAGNOSTICS (features with largest cross-cohort rank gap) ===")
    for _, r in tab.sort_values('rank_gap', ascending=False).head(4).iterrows():
        print(f"  {r['feature']:9s} rank C{r['clev_rank']:.1f}/H{r['hung_rank']:.1f} | "
              f"uniAUC C{r['clev_uniAUC']:.2f}/H{r['hung_uniAUC']:.2f} | "
              f"Hungary imputed {r['hung_imputed_%']:.1f}%")

    # extra context: age & thalach distribution by cohort
    print("\n=== POPULATION CONTEXT ===")
    for name, df in [('Cleveland', clev), ('Hungary', hung)]:
        print(f"  {name}: age {df['age'].mean():.1f}±{df['age'].std():.1f} "
              f"[{df['age'].min():.0f}-{df['age'].max():.0f}] | "
              f"thalach {df['thalach'].mean():.0f}±{df['thalach'].std():.0f} | "
              f"positives {df['num'].mean()*100:.0f}%")

    _figures(cr, hr, clev, hung, tab)
    return tab


def _figures(cr, hr, clev, hung, tab):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    color = {'strong': '#2a9d8f', 'moderate': '#e9c46a', 'weak': '#e76f51'}

    # 1. consensus importance rank per cohort, coloured by clinical strength
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=False)
    for ax, (rank, name) in zip(axes, [(cr, 'Cleveland'), (hr, 'Hungary')]):
        order = rank.sort_values(ascending=False).index          # worst rank at bottom
        vals = (len(FEATURES) + 1 - rank.loc[order])             # importance = inverse rank
        cols = [color[CLINICAL[f][1]] for f in order]
        ax.barh(range(len(order)), vals, color=cols)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        ax.set_xlabel('importance (inverse consensus rank)')
        ax.set_title(f'{name} — consensus feature importance')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    axes[1].legend(handles, [f'{k} risk factor' for k in color], loc='lower right')
    fig.suptitle('Model-important features vs established clinical strength')
    plt.tight_layout(); plt.savefig(FIGS / 'phase6_clinical_ranks.png', dpi=150); plt.close()

    # 2. univariate separability side by side
    fig, ax = plt.subplots(figsize=(9, 5))
    order = tab.sort_values('clev_uniAUC', ascending=True)['feature'].tolist()
    y = np.arange(len(order)); h = 0.38
    cvals = [tab.set_index('feature').loc[f, 'clev_uniAUC'] for f in order]
    hvals = [tab.set_index('feature').loc[f, 'hung_uniAUC'] for f in order]
    ax.barh(y + h/2, cvals, h, label='Cleveland', color='#4c72b0')
    ax.barh(y - h/2, hvals, h, label='Hungary', color='#c44e52')
    ax.axvline(0.5, ls='--', color='grey', lw=1); ax.set_xlim(0.5, 0.85)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel('single-feature separability (AUC)')
    ax.set_title('Does each feature carry disease signal, per cohort?')
    ax.legend(); plt.tight_layout(); plt.savefig(FIGS / 'phase6_univariate_auc.png', dpi=150); plt.close()

    # 3. distributions of two key divergent features by disease status, per cohort
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, feat in enumerate(['thalach', 'age']):
        for col, (name, df) in enumerate([('Cleveland', clev), ('Hungary', hung)]):
            ax = axes[row, col]
            for lab, sub in df.groupby('num'):
                ax.hist(sub[feat], bins=15, alpha=0.55,
                        label='disease' if lab == 1 else 'no disease')
            ax.set_title(f'{feat} — {name}'); ax.set_xlabel(feat); ax.legend()
    fig.suptitle('Divergent features: disease vs no-disease distributions by cohort')
    plt.tight_layout(); plt.savefig(FIGS / 'phase6_divergent_distributions.png', dpi=150); plt.close()
    print("[saved 3 figures] ->", FIGS)


if __name__ == '__main__':
    run()

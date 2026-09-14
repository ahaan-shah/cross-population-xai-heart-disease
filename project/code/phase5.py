"""
phase5.py  —  Phase 5: robust cross-population consistency analysis.

Repeats the whole explanation pipeline over many stratified splits and reports
every consistency metric as mean +/- 95% CI, with paired significance tests on:
  Q1  Is SHAP more cross-population consistent than PFI / LIME?   (per-method)
  Q2  Does adding SHAP raise the FRAMEWORK's consistency (A vs B)? (consensus)

Resumable design (so long sweeps fit in chunks):
  python phase5.py compute 0 10      # run splits 0..9, append to disk
  python phase5.py compute 10 20     # append splits 10..19
  python phase5.py compute 20 30     # append splits 20..29
  python phase5.py aggregate         # summary + tests + figures from disk
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from explain import FEATURES, SEED, fit_rf, run_pfi, _explainer
from frameworkB import shap_explain
import consistency as C

LIME_REPEATS = 1
LIME_SAMPLES = 2500
COHORTS = {'Cleveland': 'data/clean/cleveland_clean.csv',
           'Hungary':   'data/clean/hungary_clean.csv'}
METHODS = ['PFI', 'LIME', 'SHAP']
TABLES = Path('results/tables')
FIGS = Path('results/figures')
RAW = TABLES / 'phase5_raw_per_split.csv'
IMPL = TABLES / 'phase5_importances_long.csv'


def split(df, seed):
    X, y = df[FEATURES].values, df['num'].values
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


def _explain_fast(ex, x, rf, num_samples=LIME_SAMPLES):
    exp = ex.explain_instance(x, rf.predict_proba, num_features=len(FEATURES),
                              labels=(1,), num_samples=num_samples)
    d = dict(exp.as_map()[1])
    return np.array([d[i] for i in range(len(FEATURES))])


def lime_agg(rf, Xtr, Xte, n_repeats=LIME_REPEATS):
    ex = _explainer(Xtr, seed=SEED)
    per = [np.abs(np.vstack([_explain_fast(ex, x, rf) for _ in range(n_repeats)])).mean(axis=0)
           for x in Xte]
    return pd.Series(np.vstack(per).mean(axis=0), index=FEATURES)


def importances_for_split(dfs, seed):
    imp = {}
    for name, df in dfs.items():
        Xtr, Xte, ytr, yte = split(df, seed)
        rf = fit_rf(Xtr, ytr)
        pmean, _ = run_pfi(rf, Xte, yte)
        _, sglob = shap_explain(rf, Xte)
        imp[name] = {'PFI': pmean, 'LIME': lime_agg(rf, Xtr, Xte), 'SHAP': sglob}
    return imp


def consensus(imp, cohort, methods):
    ranks = [imp[cohort][m].rank(ascending=False) for m in methods]
    return sum(ranks) / len(ranks)


def metrics_for_split(imp):
    row, (C1, H1) = {}, ('Cleveland', 'Hungary')
    for m in METHODS:
        a, b = imp[C1][m], imp[H1][m]
        row[f'{m}_spearman'] = C.spearman(a, b)
        row[f'{m}_kendall'] = C.kendall(a, b)
        row[f'{m}_top3'] = C.top_k_overlap(a, b, 3)
        row[f'{m}_top5'] = C.top_k_overlap(a, b, 5)
    for fw, methods in [('A', ['PFI', 'LIME']), ('B', ['PFI', 'LIME', 'SHAP'])]:
        a, b = -consensus(imp, C1, methods), -consensus(imp, H1, methods)
        row[f'{fw}_spearman'] = C.spearman(a, b)
        row[f'{fw}_top3'] = C.top_k_overlap(a, b, 3)
        row[f'{fw}_top5'] = C.top_k_overlap(a, b, 5)
    for coh in (C1, H1):
        for x, yb in [('PFI', 'LIME'), ('PFI', 'SHAP'), ('LIME', 'SHAP')]:
            row[f'{coh[:4]}_{x}_{yb}'] = C.spearman(imp[coh][x], imp[coh][yb])
    return row


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def compute(lo, hi):
    dfs = {n: pd.read_csv(p) for n, p in COHORTS.items()}
    for s in range(lo, hi):
        imp = importances_for_split(dfs, seed=s)
        _append(RAW, [{'split': s, **metrics_for_split(imp)}])
        _append(IMPL, [{'split': s, 'cohort': c, 'method': m, 'feature': f, 'importance': v}
                       for c in COHORTS for m in METHODS for f, v in imp[c][m].items()])
        print(f"  split {s} done", flush=True)


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def aggregate():
    raw = pd.read_csv(RAW).drop_duplicates('split').sort_values('split')
    impl = pd.read_csv(IMPL).drop_duplicates(['split', 'cohort', 'method', 'feature'])
    n = len(raw)
    print(f"aggregating over {n} splits")

    summary = pd.DataFrame({
        'mean': raw.drop(columns='split').mean().round(3),
        'ci95': raw.drop(columns='split').apply(ci95).round(3),
        'std':  raw.drop(columns='split').std().round(3),
    })
    summary.to_csv(TABLES / 'phase5_consistency_summary.csv')

    mean_imp = {}
    for m in METHODS:
        for c in COHORTS:
            sub = impl[(impl.method == m) & (impl.cohort == c)]
            mean_imp[f'{m}_{c}'] = sub.groupby('feature')['importance'].mean().reindex(FEATURES)
    mean_imp = pd.DataFrame(mean_imp).round(5)
    mean_imp.to_csv(TABLES / 'phase5_mean_importances.csv')

    def paired(a, b, label):
        d = raw[b] - raw[a]
        try:
            p = wilcoxon(raw[b], raw[a]).pvalue
        except ValueError:
            p = float('nan')
        return {'comparison': label, 'mean_diff': round(d.mean(), 3),
                'ci95': round(ci95(d), 3), 'wilcoxon_p': round(p, 4),
                'significant': bool(p < 0.05) if p == p else False}
    tests = pd.DataFrame([
        paired('PFI_spearman', 'SHAP_spearman', 'SHAP vs PFI (cross-pop Spearman)'),
        paired('LIME_spearman', 'SHAP_spearman', 'SHAP vs LIME (cross-pop Spearman)'),
        paired('A_spearman', 'B_spearman', 'Framework B vs A (consensus Spearman)'),
    ])
    tests.to_csv(TABLES / 'phase5_headtohead_tests.csv', index=False)

    print(f"\n=== CROSS-POPULATION CONSISTENCY (mean +/- 95pct CI, n={n}) ===")
    for m in METHODS:
        print(f"  {m:5s} Spearman {summary.loc[m+'_spearman','mean']:.3f} "
              f"+/- {summary.loc[m+'_spearman','ci95']:.3f}   "
              f"top3 {summary.loc[m+'_top3','mean']:.2f}/3   top5 {summary.loc[m+'_top5','mean']:.2f}/5")
    print(f"  A     Spearman {summary.loc['A_spearman','mean']:.3f} +/- {summary.loc['A_spearman','ci95']:.3f}")
    print(f"  B     Spearman {summary.loc['B_spearman','mean']:.3f} +/- {summary.loc['B_spearman','ci95']:.3f}")
    print("\n=== HEAD-TO-HEAD PAIRED WILCOXON TESTS ===\n", tests.to_string(index=False))

    _figures(raw, summary, mean_imp)
    return summary, tests


def _figures(raw, summary, mean_imp):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [summary.loc[m+'_spearman', 'mean'] for m in METHODS]
    cis = [summary.loc[m+'_spearman', 'ci95'] for m in METHODS]
    ax.bar(METHODS, means, yerr=cis, capsize=5, color=['#8c8c8c', '#dd8452', '#4c72b0'])
    ax.set_ylabel('cross-population Spearman'); ax.set_ylim(0, 1)
    ax.set_title('Cross-population consistency by method (mean +/- 95% CI)')
    for i, m in enumerate(means):
        ax.text(i, m+cis[i]+0.02, f'{m:.2f}', ha='center')
    plt.tight_layout(); plt.savefig(FIGS / 'phase5_per_method_crosspop.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([raw['A_spearman'], raw['B_spearman']],
               tick_labels=['Framework A\n(PFI+LIME)', 'Framework B\n(+SHAP)'])
    ax.set_ylabel('consensus cross-population Spearman')
    ax.set_title('Framework A vs B — consensus consistency across splits')
    plt.tight_layout(); plt.savefig(FIGS / 'phase5_framework_AvB.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([raw[m+"_spearman"] for m in METHODS], tick_labels=METHODS)
    ax.set_ylabel('cross-population Spearman (per split)')
    ax.set_title('Spread of cross-population consistency across splits')
    plt.tight_layout(); plt.savefig(FIGS / 'phase5_crosspop_distributions.png', dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, coh in zip(axes, ['Clev', 'Hung']):
        M = pd.DataFrame(np.eye(3), index=METHODS, columns=METHODS)
        for x, yb in [('PFI', 'LIME'), ('PFI', 'SHAP'), ('LIME', 'SHAP')]:
            v = summary.loc[f'{coh}_{x}_{yb}', 'mean']; M.loc[x, yb] = M.loc[yb, x] = v
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap='YlGnBu')
        ax.set_xticks(range(3)); ax.set_xticklabels(METHODS)
        ax.set_yticks(range(3)); ax.set_yticklabels(METHODS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{M.values[i,j]:.2f}', ha='center', va='center',
                        color='white' if M.values[i, j] > 0.6 else 'black')
        ax.set_title('Cleveland' if coh == 'Clev' else 'Hungary')
    fig.suptitle('Cross-method agreement (mean Spearman over splits)')
    fig.colorbar(im, ax=axes, fraction=0.025)
    plt.savefig(FIGS / 'phase5_crossmethod_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, m in zip(axes, METHODS):
        t = pd.DataFrame({'Cleveland': mean_imp[f'{m}_Cleveland'],
                          'Hungary': mean_imp[f'{m}_Hungary']})
        order = t.mean(axis=1).sort_values().index; t = t.loc[order]
        y = np.arange(len(t)); h = 0.38
        ax.barh(y+h/2, t['Cleveland'], h, label='Cleveland', color='#4c72b0')
        ax.barh(y-h/2, t['Hungary'], h, label='Hungary', color='#c44e52')
        ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_title(m); ax.set_xlabel('mean importance')
    axes[0].legend()
    fig.suptitle('Feature importance by method — Cleveland vs Hungary (mean over splits)')
    plt.tight_layout(); plt.savefig(FIGS / 'phase5_ranking_sidebyside.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = METHODS + ['A', 'B']; x = np.arange(len(groups)); w = 0.38
    t3 = [summary.loc[g+'_top3', 'mean'] for g in groups]
    t5 = [summary.loc[g+'_top5', 'mean'] for g in groups]
    ax.bar(x-w/2, t3, w, label='top-3 (max 3)', color='#4c72b0')
    ax.bar(x+w/2, t5, w, label='top-5 (max 5)', color='#dd8452')
    ax.set_xticks(x); ax.set_xticklabels(['PFI', 'LIME', 'SHAP', 'Fwk A', 'Fwk B'])
    ax.set_ylabel('mean shared features'); ax.set_title('Top-k cross-population overlap'); ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / 'phase5_topk_overlap.png', dpi=150); plt.close()
    print("[saved 6 figures] ->", FIGS)


USAGE = """usage:
  python code/phase5.py compute <lo> <hi>   # run splits lo..hi-1, append to disk
  python code/phase5.py aggregate          # summary + tests + figures from disk"""


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'compute':
        compute(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == 'aggregate':
        aggregate()
    else:
        sys.exit(USAGE)

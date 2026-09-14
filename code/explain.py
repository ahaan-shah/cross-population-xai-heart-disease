"""
explain.py  —  Phase 3: Framework A (LIME + PFI) on the chosen model (Random Forest).

Produces, for each cohort:
  * PFI global feature-importance ranking (permutation importance, AUC-scored, on the test set).
  * A LIME stability characterisation (each of several representative patients explained
    many times; we measure how much the explanation wobbles across reruns).
  * An aggregated LIME importance ranking (mean |weight| per feature over test patients,
    each averaged over repeats to denoise).
Then computes Framework-A consistency numbers:
  * cross-population  (Cleveland vs Hungary) for PFI and for LIME
  * cross-method      (PFI vs LIME) within each cohort

Design choices (documented for the report):
  * RF is scale-invariant, so explanations are done on ORIGINAL (unscaled) feature units,
    which keeps LIME/PFI interpretable in clinical terms (mmHg, bpm, ...).
  * 80/20 stratified split (seed 42); RF trained on train; PFI + LIME evaluated on the
    held-out test set (importance that generalises, instances the model did not see).
  * LIME: discretize_continuous=True (default; interpretable "feature-in-range" weights).
    Explanations are for label 1 (disease), i.e. "contribution toward a disease prediction".
  * All randomness seeded -> deterministic, reproducible outputs.

Run:  python explain.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from lime.lime_tabular import LimeTabularExplainer

import consistency as C

SEED = 42
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak']
N_STAB_REPEATS = 30      # reruns per patient for the stability protocol
N_STAB_PATIENTS = 6      # representative patients (2 disease, 2 healthy, 2 borderline)
N_AGG_REPEATS = 5        # reruns per patient, averaged, for the aggregate ranking
PFI_REPEATS = 30


# --------------------------------------------------------------------------- #
# data & model
# --------------------------------------------------------------------------- #
def load_split(path):
    df = pd.read_csv(path)
    X, y = df[FEATURES], df['num'].values
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.20, stratify=y, random_state=SEED)
    return Xtr, Xte, ytr, yte


def fit_rf(Xtr, ytr):
    return RandomForestClassifier(random_state=SEED).fit(Xtr, ytr)


# --------------------------------------------------------------------------- #
# PFI
# --------------------------------------------------------------------------- #
def run_pfi(rf, Xte, yte):
    r = permutation_importance(rf, Xte, yte, scoring='roc_auc',
                               n_repeats=PFI_REPEATS, random_state=SEED)
    mean = pd.Series(r.importances_mean, index=FEATURES)
    std = pd.Series(r.importances_std, index=FEATURES)
    return mean, std


# --------------------------------------------------------------------------- #
# LIME helpers
# --------------------------------------------------------------------------- #
def _explainer(Xtr, seed=SEED):
    return LimeTabularExplainer(
        Xtr, feature_names=FEATURES, class_names=['no_disease', 'disease'],
        mode='classification', discretize_continuous=True, random_state=seed)


def _weights(exp):
    """Return a length-10 vector of LIME weights (feature order) for label 'disease'."""
    d = dict(exp.as_map()[1])
    return np.array([d[i] for i in range(len(FEATURES))])


def _explain_once(explainer, x, rf):
    exp = explainer.explain_instance(x, rf.predict_proba,
                                     num_features=len(FEATURES), labels=(1,))
    return _weights(exp)


def select_patients(rf, Xte):
    """2 most-confident disease, 2 most-confident healthy, 2 borderline (by predicted proba)."""
    p = rf.predict_proba(Xte)[:, 1]
    order = np.argsort(p)
    borderline = np.argsort(np.abs(p - 0.5))[:2]
    return list(order[-2:]) + list(order[:2]) + list(borderline)


# --------------------------------------------------------------------------- #
# LIME stability protocol
# --------------------------------------------------------------------------- #
def lime_stability(rf, Xtr, Xte):
    """Explain each representative patient N times; quantify rerun wobble."""
    patients = select_patients(rf, Xte)
    per_patient_rank_stab, per_patient_top3_jac, per_patient_wstd = [], [], []
    raw_for_fig = None
    for pi, idx in enumerate(patients):
        ex = _explainer(Xtr, seed=SEED)
        W = np.vstack([_explain_once(ex, Xte[idx], rf) for _ in range(N_STAB_REPEATS)])  # N x 10
        # rank stability: mean pairwise Spearman of |weights| across reruns
        from scipy.stats import spearmanr
        rho = [spearmanr(np.abs(W[i]), np.abs(W[j])).correlation
               for i in range(len(W)) for j in range(i + 1, len(W))]
        per_patient_rank_stab.append(np.nanmean(rho))
        # top-3 stability: mean pairwise Jaccard of top-3 |weight| features across reruns
        tops = [set(np.argsort(np.abs(W[i]))[-3:]) for i in range(len(W))]
        jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j])
               for i in range(len(tops)) for j in range(i + 1, len(tops))]
        per_patient_top3_jac.append(np.mean(jac))
        # weight wobble: mean over features of the std of the weight across reruns
        per_patient_wstd.append(W.std(axis=0).mean())
        if pi == len(patients) - 1:   # keep the last (a borderline) patient for the figure
            raw_for_fig = W
    return {
        'rank_stability_spearman': round(float(np.mean(per_patient_rank_stab)), 3),
        'top3_jaccard':            round(float(np.mean(per_patient_top3_jac)), 3),
        'mean_weight_std':         round(float(np.mean(per_patient_wstd)), 4),
    }, raw_for_fig


# --------------------------------------------------------------------------- #
# LIME aggregate importance
# --------------------------------------------------------------------------- #
def lime_aggregate(rf, Xtr, Xte):
    """Mean |LIME weight| per feature over all test patients (each averaged over repeats)."""
    ex = _explainer(Xtr, seed=SEED)
    per_patient = []
    for x in Xte:
        reps = np.vstack([_explain_once(ex, x, rf) for _ in range(N_AGG_REPEATS)])
        per_patient.append(np.abs(reps).mean(axis=0))       # mean |w| over repeats
    agg = np.vstack(per_patient).mean(axis=0)               # mean over patients
    return pd.Series(agg, index=FEATURES)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(clean_dir='data/clean', out_tables='results/tables', out_figs='results/figures'):
    out_tables, out_figs = Path(out_tables), Path(out_figs)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    cohorts = {'Cleveland': f'{clean_dir}/cleveland_clean.csv',
               'Hungary':   f'{clean_dir}/hungary_clean.csv'}
    pfi_imp, lime_imp, stab, figdata = {}, {}, {}, {}

    for name, path in cohorts.items():
        Xtr, Xte, ytr, yte = load_split(path)
        rf = fit_rf(Xtr, ytr)
        auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
        pmean, pstd = run_pfi(rf, Xte, yte)
        pfi_imp[name] = pmean
        stab[name], figdata[name] = lime_stability(rf, Xtr, Xte)
        lime_imp[name] = lime_aggregate(rf, Xtr, Xte)
        print(f"[{name}] test AUC={auc:.3f} | LIME rank-stability(Spearman)="
              f"{stab[name]['rank_stability_spearman']} top3-Jaccard={stab[name]['top3_jaccard']}")

    # ---- assemble importance tables ----
    pfi_tab = pd.DataFrame({k: v for k, v in pfi_imp.items()}).round(4)
    lime_tab = pd.DataFrame({k: v for k, v in lime_imp.items()}).round(5)
    pfi_tab.to_csv(out_tables / 'frameworkA_pfi.csv')
    lime_tab.to_csv(out_tables / 'frameworkA_lime_importance.csv')
    pd.DataFrame(stab).T.to_csv(out_tables / 'frameworkA_lime_stability.csv')

    print("\n=== PFI importance (AUC-drop) ===\n", pfi_tab.to_string())
    print("\n=== Aggregated LIME importance (mean |weight|) ===\n", lime_tab.to_string())

    # ---- consistency numbers ----
    rows = []
    rows.append(('PFI',  'cross-population (Clev vs Hung)',
                 C.compare(pfi_imp['Cleveland'], pfi_imp['Hungary'])))
    rows.append(('LIME', 'cross-population (Clev vs Hung)',
                 C.compare(lime_imp['Cleveland'], lime_imp['Hungary'])))
    rows.append(('PFI vs LIME', 'cross-method @ Cleveland',
                 C.compare(pfi_imp['Cleveland'], lime_imp['Cleveland'])))
    rows.append(('PFI vs LIME', 'cross-method @ Hungary',
                 C.compare(pfi_imp['Hungary'], lime_imp['Hungary'])))
    cons = pd.DataFrame([{'method': m, 'comparison': c, **vals} for m, c, vals in rows])
    cons.to_csv(out_tables / 'frameworkA_consistency.csv', index=False)
    print("\n=== FRAMEWORK A CONSISTENCY ===\n", cons.to_string(index=False))
    print("\n=== LIME STABILITY (noise floor) ===\n", pd.DataFrame(stab).T.to_string())

    _figures(pfi_tab, lime_tab, figdata, out_figs)
    return pfi_tab, lime_tab, stab, cons


def _figures(pfi_tab, lime_tab, figdata, out_figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def ranking_fig(tab, title, fname, xlabel):
        order = tab.mean(axis=1).sort_values().index
        t = tab.loc[order]
        y = np.arange(len(t)); h = 0.38
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(y + h/2, t['Cleveland'], h, label='Cleveland', color='#4C72B0')
        ax.barh(y - h/2, t['Hungary'],  h, label='Hungary',  color='#C44E52')
        ax.set_yticks(y); ax.set_yticklabels(t.index)
        ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()
        plt.tight_layout(); plt.savefig(out_figs / fname, dpi=150, bbox_inches='tight'); plt.close()

    ranking_fig(pfi_tab, 'Framework A — PFI importance (RF, AUC-drop)',
                'frameworkA_pfi.png', 'permutation importance (AUC drop)')
    ranking_fig(lime_tab, 'Framework A — aggregated LIME importance (RF)',
                'frameworkA_lime_importance.png', 'mean |LIME weight|')

    # LIME instability illustration: weight spread across 30 reruns for one borderline patient
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, W) in zip(axes, figdata.items()):
        ax.boxplot([W[:, i] for i in range(len(FEATURES))], labels=FEATURES, vert=True)
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_title(f'LIME weight spread over 30 reruns — {name}\n(one borderline patient)')
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('LIME weight (toward disease)')
    plt.tight_layout(); plt.savefig(out_figs / 'frameworkA_lime_stability.png',
                                    dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n[saved figures] -> {out_figs}")


if __name__ == '__main__':
    run()

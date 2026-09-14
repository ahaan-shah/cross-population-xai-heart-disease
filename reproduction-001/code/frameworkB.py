"""
frameworkB.py  —  Phase 4: Framework B = SHAP + LIME + PFI on Random Forest.

Adds SHAP (exact, deterministic TreeExplainer) to the LIME+PFI pair from Phase 3,
on the SAME 80/20 split and RF model, and recomputes consistency for the trio.

Produces per cohort:
  * SHAP global importance  (mean |SHAP| for the disease class, over the test set)
  * SHAP local attributions (waterfall plots for representative patients; beeswarm)
Then, using the Phase-3 PFI and LIME importances (loaded from results/tables):
  * cross-population consistency of SHAP (Cleveland vs Hungary)
  * cross-method agreement among PFI / LIME / SHAP within each cohort
  * a preview of the Framework A vs B head-to-head via a consensus ranking
    (Framework A = mean rank of PFI+LIME; Framework B = mean rank of PFI+LIME+SHAP)

Depends on Phase 3 outputs (results/tables/frameworkA_pfi.csv, frameworkA_lime_importance.csv).
Run:  python frameworkB.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from explain import load_split, fit_rf, select_patients, FEATURES, SEED
import consistency as C

COHORTS = {'Cleveland': 'data/clean/cleveland_clean.csv',
           'Hungary':   'data/clean/hungary_clean.csv'}


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_explain(rf, Xte):
    """Return (class-1 Explanation, global mean|SHAP| Series)."""
    explainer = shap.TreeExplainer(rf)
    e = explainer(Xte)                       # values: (n, features, 2 classes)
    expl1 = shap.Explanation(
        values=e.values[:, :, 1],            # disease class
        base_values=e.base_values[:, 1],
        data=Xte, feature_names=FEATURES)
    glob = pd.Series(np.abs(expl1.values).mean(axis=0), index=FEATURES)
    return expl1, glob


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _beeswarm(expl1, cohort, path):
    shap.plots.beeswarm(expl1, show=False, max_display=10)
    plt.title(f'SHAP summary (beeswarm) — {cohort}')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def _waterfall(expl1, i, cohort, label, path):
    shap.plots.waterfall(expl1[i], show=False, max_display=10)
    plt.title(f'SHAP local explanation — {cohort} ({label})')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def _global_bar(shap_tab, path):
    order = shap_tab.mean(axis=1).sort_values().index
    t = shap_tab.loc[order]; y = np.arange(len(t)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + h/2, t['Cleveland'], h, label='Cleveland', color='#4C72B0')
    ax.barh(y - h/2, t['Hungary'],  h, label='Hungary',  color='#C44E52')
    ax.set_yticks(y); ax.set_yticklabels(t.index)
    ax.set_xlabel('mean |SHAP| (disease class)')
    ax.set_title('Framework B — SHAP global importance (RF)'); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def _crossmethod_heatmap(mats, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (cohort, M) in zip(axes, mats.items()):
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap='YlGnBu')
        ax.set_xticks(range(3)); ax.set_xticklabels(M.columns)
        ax.set_yticks(range(3)); ax.set_yticklabels(M.index)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{M.values[i,j]:.2f}', ha='center', va='center',
                        color='white' if M.values[i, j] > 0.6 else 'black')
        ax.set_title(f'Cross-method agreement (Spearman)\n{cohort}')
    fig.colorbar(im, ax=axes, fraction=0.025)
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(out_tables='results/tables', out_figs='results/figures'):
    out_tables, out_figs = Path(out_tables), Path(out_figs)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Phase-3 importances (deterministic, canonical)
    pfi = pd.read_csv(out_tables / 'frameworkA_pfi.csv', index_col=0)
    lime = pd.read_csv(out_tables / 'frameworkA_lime_importance.csv', index_col=0)

    shap_imp, local_rows = {}, []
    for name, path in COHORTS.items():
        Xtr, Xte, ytr, yte = load_split(path)
        rf = fit_rf(Xtr, ytr)
        expl1, glob = shap_explain(rf, Xte)
        shap_imp[name] = glob

        # local: one confident-disease and one confident-healthy patient
        pats = select_patients(rf, Xte)          # [2 disease, 2 healthy, 2 borderline]
        p = rf.predict_proba(Xte)[:, 1]
        i_dis, i_hlt = pats[0], pats[2]
        _beeswarm(expl1, name, out_figs / f'frameworkB_shap_beeswarm_{name.lower()}.png')
        _waterfall(expl1, i_dis, name, f'disease case, p={p[i_dis]:.2f}',
                   out_figs / f'frameworkB_shap_waterfall_{name.lower()}_disease.png')
        _waterfall(expl1, i_hlt, name, f'healthy case, p={p[i_hlt]:.2f}',
                   out_figs / f'frameworkB_shap_waterfall_{name.lower()}_healthy.png')
        for lbl, idx in [('disease', i_dis), ('healthy', i_hlt)]:
            row = {'cohort': name, 'case': lbl, 'p_disease': round(float(p[idx]), 3)}
            row.update({f: round(float(v), 4) for f, v in zip(FEATURES, expl1.values[idx])})
            local_rows.append(row)

    shap_tab = pd.DataFrame(shap_imp).round(5)
    shap_tab.to_csv(out_tables / 'frameworkB_shap_global.csv')
    pd.DataFrame(local_rows).to_csv(out_tables / 'frameworkB_shap_local_examples.csv', index=False)
    print("=== SHAP global importance (mean |SHAP|, disease class) ===\n", shap_tab.to_string())

    # -------- cross-population consistency of SHAP --------
    shap_cons = C.compare(shap_imp['Cleveland'], shap_imp['Hungary'])

    # -------- cross-method agreement within each cohort (PFI/LIME/SHAP) --------
    mats = {}
    for name in COHORTS:
        vecs = {'PFI': pfi[name], 'LIME': lime[name], 'SHAP': shap_imp[name]}
        M = pd.DataFrame(index=vecs, columns=vecs, dtype=float)
        for a in vecs:
            for b in vecs:
                M.loc[a, b] = 1.0 if a == b else round(C.spearman(vecs[a], vecs[b]), 3)
        mats[name] = M
        M.to_csv(out_tables / f'frameworkB_crossmethod_{name.lower()}.csv')

    # -------- consensus ranking: Framework A (PFI+LIME) vs B (PFI+LIME+SHAP) --------
    def rank_desc(s):                      # importance -> rank (1 = most important)
        return s.rank(ascending=False)

    consensusA, consensusB = {}, {}
    for name in COHORTS:
        rp, rl, rs = rank_desc(pfi[name]), rank_desc(lime[name]), rank_desc(shap_imp[name])
        consensusA[name] = (rp + rl) / 2
        consensusB[name] = (rp + rl + rs) / 3
    # convert avg-rank to importance-like (higher = more important) for compare()
    A_cons = C.compare(-consensusA['Cleveland'], -consensusA['Hungary'])
    B_cons = C.compare(-consensusB['Cleveland'], -consensusB['Hungary'])

    # -------- assemble consistency table --------
    cons = pd.DataFrame([
        {'row': 'SHAP — cross-population', **shap_cons},
        {'row': 'Framework A consensus — cross-population', **A_cons},
        {'row': 'Framework B consensus — cross-population', **B_cons},
    ])
    cons.to_csv(out_tables / 'frameworkB_consistency.csv', index=False)

    print("\n=== SHAP cross-population consistency ===")
    print(pd.Series(shap_cons).to_string())
    print("\n=== Cross-method agreement (Spearman) ===")
    for name, M in mats.items():
        print(f"\n[{name}]\n{M.to_string()}")
    print("\n=== FRAMEWORK A vs B — consensus cross-population consistency ===")
    print(cons.set_index('row').to_string())

    _global_bar(shap_tab, out_figs / 'frameworkB_shap_global.png')
    _crossmethod_heatmap(mats, out_figs / 'frameworkB_crossmethod_heatmap.png')
    print(f"\n[saved figures] -> {out_figs}")
    return shap_tab, cons, mats


if __name__ == '__main__':
    run()

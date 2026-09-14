"""
models.py  —  Phase 2: train & evaluate the 7 classifiers on each cohort.

For each cohort we run stratified k-fold cross-validation and report five metrics
(accuracy, precision, recall, F1, AUC-ROC) as mean ± std across folds, then rank
the models to select a single architecture to carry into the XAI phases.

Leakage control
---------------
Scale-sensitive models (LR, SVM, KNN) are wrapped in a Pipeline(StandardScaler, clf)
so the scaler is fit on the training folds ONLY, never on the held-out fold.
Tree models (DT, RF, XGBoost) and Gaussian NB are scale-invariant and used directly.

Fold shuffling
--------------
StratifiedKFold(shuffle=True) is REQUIRED here: the Hungarian file is sorted by age,
so unshuffled folds would be age-biased. The seed makes the split reproducible.

Positive class = 1 (disease); precision/recall/F1 are reported for that class,
which is the clinically important one (catching disease).

Run:  python models.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

SEED = 42
K = 10  # stratified k-fold

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak']

# display / iteration order (matches the project docs)
MODEL_ORDER = ['LR', 'SVM', 'DT', 'RF', 'XGBoost', 'NB', 'KNN']
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
SCALE_SENSITIVE = {'LR', 'SVM', 'KNN'}

# --- model-selection rule (see apply_selection_rule) ---------------------------- #
# Models within TIE_TOL of the best mean AUC are treated as statistically tied: the
# top four span 0.007 AUC against a fold std of ~0.05, so their ordering is noise.
TIE_TOL = 0.01
NONLINEAR = {'SVM', 'DT', 'RF', 'XGBoost', 'KNN'}   # LIME needs something to approximate
TREE_BASED = {'DT', 'RF', 'XGBoost'}                # unlocks SHAP's exact TreeExplainer
EXPLAINED_MODEL = 'RF'                              # what the XAI phases are built on


def build_models():
    """Return {name: estimator}. Scale-sensitive models get an in-pipeline scaler."""
    base = {
        'LR':      LogisticRegression(max_iter=1000, random_state=SEED),
        'SVM':     SVC(probability=True, random_state=SEED),
        'DT':      DecisionTreeClassifier(random_state=SEED),
        'RF':      RandomForestClassifier(random_state=SEED),
        'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=1),
        'NB':      GaussianNB(),
        'KNN':     KNeighborsClassifier(),
    }
    out = {}
    for name, clf in base.items():
        out[name] = make_pipeline(StandardScaler(), clf) if name in SCALE_SENSITIVE else clf
    return out


SCORING = {
    'accuracy':  'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall':    make_scorer(recall_score, zero_division=0),
    'f1':        make_scorer(f1_score, zero_division=0),
    'roc_auc':   'roc_auc',
}


def evaluate_cohort(df: pd.DataFrame, cohort: str, k: int = K) -> pd.DataFrame:
    """Run k-fold CV for all 7 models on one cohort; return a mean/std metric table."""
    X = df[FEATURES].values
    y = df['num'].values
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    rows = []
    for name in MODEL_ORDER:
        model = build_models()[name]
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {'model': name}
        for m in METRICS:
            row[f'{m}_mean'] = res[f'test_{m}'].mean()
            row[f'{m}_std']  = res[f'test_{m}'].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')


def format_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw mean/std frame into a display table of 'mean ± std' strings."""
    disp = pd.DataFrame(index=metric_df.index)
    pretty = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
              'f1': 'F1', 'roc_auc': 'AUC-ROC'}
    for m in METRICS:
        disp[pretty[m]] = [f"{mu:.3f} ± {sd:.3f}"
                           for mu, sd in zip(metric_df[f'{m}_mean'], metric_df[f'{m}_std'])]
    return disp


def select_model(clev: pd.DataFrame, hung: pd.DataFrame) -> pd.DataFrame:
    """Rank models by AUC-ROC and F1 averaged across the two cohorts."""
    rank = pd.DataFrame(index=MODEL_ORDER)
    rank['AUC_cleveland'] = clev['roc_auc_mean']
    rank['AUC_hungary']   = hung['roc_auc_mean']
    rank['AUC_avg']       = rank[['AUC_cleveland', 'AUC_hungary']].mean(axis=1)
    rank['F1_cleveland']  = clev['f1_mean']
    rank['F1_hungary']    = hung['f1_mean']
    rank['F1_avg']        = rank[['F1_cleveland', 'F1_hungary']].mean(axis=1)
    return rank.sort_values('AUC_avg', ascending=False).round(3)


def apply_selection_rule(ranking: pd.DataFrame):
    """Apply the selection rule stated in advance, and return (winner, tied, chosen).

    The rule: *best mean AUC across both cohorts; ties broken on fitness-for-purpose
    and explainer compatibility.*

    Ranking by AUC alone is NOT sufficient here and must not be used on its own: the
    top four models are separated by 0.007 AUC against a fold std of about 0.05, so
    they are statistically tied and the AUC ordering among them is noise. The tie-break
    is what actually decides, on two grounds:

      * The model must be NONLINEAR. LIME fits a linear local surrogate, so on an
        already-linear model that approximation is near-exact and LIME, SHAP and PFI
        would agree trivially -- the Framework A vs B comparison would have nothing
        left to discriminate. Choosing the top-ranked LR would hollow out the project.
      * A TREE model unlocks SHAP's TreeExplainer: exact, fast and deterministic,
        which is the direct methodological contrast to LIME's stochasticity.

    Random Forest is the only model in the tied band meeting both, and it is also the
    single best model on Hungary (0.891). Recorded caveat: its default-threshold F1
    (0.727) trails LR/NB (~0.76); selection is AUC-led (threshold-independent) and the
    purpose is explanation, not deployment.
    """
    winner = ranking.index[0]                       # best raw AUC_avg
    best = ranking.loc[winner, 'AUC_avg']
    tied = ranking.index[ranking['AUC_avg'] >= best - TIE_TOL].tolist()

    eligible = [m for m in tied if m in NONLINEAR and m in TREE_BASED]
    chosen = eligible[0] if eligible else winner    # `tied` is already AUC-sorted

    assert chosen == EXPLAINED_MODEL, (
        f"selection rule chose {chosen}, but the whole XAI pipeline is built on "
        f"{EXPLAINED_MODEL} (explain.fit_rf). Re-decide the rule before proceeding: "
        f"tied band was {tied}.")
    return winner, tied, chosen


def run(clean_dir='data/clean', out_tables='results/tables', out_figs='results/figures'):
    out_tables, out_figs = Path(out_tables), Path(out_figs)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    clev_df = pd.read_csv(Path(clean_dir) / 'cleveland_clean.csv')
    hung_df = pd.read_csv(Path(clean_dir) / 'hungary_clean.csv')

    clev = evaluate_cohort(clev_df, 'Cleveland')
    hung = evaluate_cohort(hung_df, 'Hungary')

    # save raw + formatted tables
    clev.to_csv(out_tables / 'cleveland_metrics_raw.csv')
    hung.to_csv(out_tables / 'hungary_metrics_raw.csv')
    format_table(clev).to_csv(out_tables / 'cleveland_metrics.csv')
    format_table(hung).to_csv(out_tables / 'hungary_metrics.csv')

    print("=== CLEVELAND (stratified 10-fold CV) ===")
    print(format_table(clev).to_string(), "\n")
    print("=== HUNGARY (stratified 10-fold CV) ===")
    print(format_table(hung).to_string(), "\n")

    ranking = select_model(clev, hung)
    ranking.to_csv(out_tables / 'model_selection_ranking.csv')
    print("=== MODEL SELECTION RANKING (by mean AUC-ROC across both cohorts) ===")
    print(ranking.to_string(), "\n")
    winner, tied, chosen = apply_selection_rule(ranking)
    print("=== SELECTION RULE: best mean AUC across both cohorts; "
          "ties broken on fitness-for-purpose and explainer compatibility ===")
    print(f"  highest mean AUC      : {winner} ({ranking.loc[winner, 'AUC_avg']})")
    print(f"  statistically tied    : {tied}  (within {TIE_TOL} AUC of the best, "
          f"against a fold std of ~0.05)")
    print(f"  tie-break             : must be nonlinear (LIME fits a linear local "
          f"surrogate) AND tree-based (exact, deterministic SHAP TreeExplainer)")
    print(f">>> Selected architecture: {chosen} "
          f"(AUC_avg={ranking.loc[chosen, 'AUC_avg']}, F1_avg={ranking.loc[chosen, 'F1_avg']}, "
          f"best on Hungary at {ranking.loc[chosen, 'AUC_hungary']})")
    if chosen != winner:
        print(f"    NOTE: {chosen} is not the top-ranked model. {winner} leads on raw AUC by "
              f"{ranking.loc[winner, 'AUC_avg'] - ranking.loc[chosen, 'AUC_avg']:.3f}, which is "
              f"an order of magnitude inside the noise. Selecting the linear {winner} would "
              f"make LIME/SHAP/PFI agree trivially and leave the A-vs-B comparison nothing "
              f"to discriminate.")

    decision = pd.DataFrame([{
        'rule': 'best mean AUC across both cohorts; ties broken on fitness-for-purpose '
                'and explainer compatibility',
        'highest_mean_AUC': winner,
        'tie_tolerance': TIE_TOL,
        'statistically_tied': ' '.join(tied),
        'tiebreak': 'nonlinear (LIME surrogate must have something to approximate) '
                    'AND tree-based (exact deterministic SHAP TreeExplainer)',
        'selected': chosen,
        'selected_AUC_avg': ranking.loc[chosen, 'AUC_avg'],
        'selected_AUC_hungary': ranking.loc[chosen, 'AUC_hungary'],
        'selected_F1_avg': ranking.loc[chosen, 'F1_avg'],
        'caveat': f"default-threshold F1 {ranking.loc[chosen, 'F1_avg']} trails "
                  f"LR/NB (~0.76); selection is AUC-led and threshold-independent, "
                  f"and the purpose is explanation, not deployment",
    }])
    decision.to_csv(out_tables / 'model_selection_decision.csv', index=False)

    _make_figure(clev, hung, out_figs / 'model_comparison.png')
    return clev, hung, ranking


def _make_figure(clev, hung, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(MODEL_ORDER)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, metric, title in [(axes[0], 'roc_auc', 'AUC-ROC'), (axes[1], 'f1', 'F1')]:
        ax.bar(x - w/2, clev[f'{metric}_mean'], w, yerr=clev[f'{metric}_std'],
               capsize=3, label='Cleveland', color='#4C72B0')
        ax.bar(x + w/2, hung[f'{metric}_mean'], w, yerr=hung[f'{metric}_std'],
               capsize=3, label='Hungary', color='#C44E52')
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=0)
        ax.set_title(f'{title} by model (10-fold CV, mean ± std)')
        ax.set_ylim(0, 1.0); ax.grid(axis='y', alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[saved] {path}")


if __name__ == '__main__':
    run()

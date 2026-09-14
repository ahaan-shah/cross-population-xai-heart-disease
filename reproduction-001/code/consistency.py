"""
consistency.py  —  reusable explanation-consistency metrics.

Every XAI method in this project ultimately produces one importance value per
feature. These helpers compare two such importance vectors (e.g. Cleveland vs
Hungary, or PFI vs LIME) and quantify how much they agree.

Used in Phase 3 (Framework A) and again in Phase 5 (full cross-framework analysis).

An "importance vector" is a pandas Series indexed by feature name.
Rank-based metrics (Spearman, Kendall) are scale-free, so raw importance values
from different methods can be compared without normalisation.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def _align(a: pd.Series, b: pd.Series):
    """Restrict two importance Series to their shared features, same order."""
    idx = [f for f in a.index if f in b.index]
    return a.loc[idx], b.loc[idx]


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation of two importance vectors (+1 = identical order)."""
    a, b = _align(a, b)
    return float(spearmanr(a.values, b.values).correlation)


def kendall(a: pd.Series, b: pd.Series) -> float:
    """Kendall's tau of two importance vectors (robust rank-agreement)."""
    a, b = _align(a, b)
    return float(kendalltau(a.values, b.values).correlation)


def top_k(s: pd.Series, k: int) -> list:
    """The k most important feature names (descending)."""
    return list(s.sort_values(ascending=False).index[:k])


def top_k_overlap(a: pd.Series, b: pd.Series, k: int) -> int:
    """How many features appear in BOTH top-k lists (0..k)."""
    return len(set(top_k(a, k)) & set(top_k(b, k)))


def jaccard_top_k(a: pd.Series, b: pd.Series, k: int) -> float:
    """Jaccard overlap of the two top-k sets: |A∩B| / |A∪B|."""
    A, B = set(top_k(a, k)), set(top_k(b, k))
    return len(A & B) / len(A | B)


def compare(a: pd.Series, b: pd.Series, ks=(3, 5)) -> dict:
    """Full consistency report between two importance vectors."""
    out = {'spearman': round(spearman(a, b), 3),
           'kendall':  round(kendall(a, b), 3)}
    for k in ks:
        out[f'top{k}_overlap'] = f"{top_k_overlap(a, b, k)}/{k}"
        out[f'top{k}_jaccard'] = round(jaccard_top_k(a, b, k), 3)
    return out

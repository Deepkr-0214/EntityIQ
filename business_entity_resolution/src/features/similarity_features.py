"""
Similarity feature engineering module.

Computes pairwise features between S1 entities and their candidates.
All functions accept plain strings or lists — no DataFrame dependency.
"""
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


# ══════════════════════════════════════════════════════════════════════════════
#  Atomic similarity helpers
# ══════════════════════════════════════════════════════════════════════════════

def jaccard_token(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def norm_lev(a: str, b: str) -> float:
    """Normalised Levenshtein similarity = 1 − (dist / max_len)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ml = max(len(a), len(b))
    return 1.0 - Levenshtein.distance(a, b) / ml


def tok_sort(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def tok_set(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def partial(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def wratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b) / 100.0


def monge_elkan(a: str, b: str) -> float:
    """For each token in A find best match in B, average the best scores."""
    if not a or not b:
        return 0.0
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    total = 0.0
    for wa in ta:
        best = max(fuzz.ratio(wa, wb) for wb in tb) / 100.0
        total += best
    return total / len(ta)


def acronym_match(a: str, b: str) -> float:
    """Check if one name's initials spell the other."""
    if not a or not b:
        return 0.0
    ta, tb = a.split(), b.split()
    # A is potential acronym of B
    if len(ta) == 1 and len(ta[0]) >= 2:
        initials = ''.join(w[0] for w in tb if w)
        if ta[0] == initials:
            return 1.0
    # B is potential acronym of A
    if len(tb) == 1 and len(tb[0]) >= 2:
        initials = ''.join(w[0] for w in ta if w)
        if tb[0] == initials:
            return 1.0
    return 0.0


def numeric_features(nums_a: list, nums_b: list):
    """Returns (any_match, fraction_matched, longest_match_length)."""
    if not nums_a or not nums_b:
        return 0, 0.0, 0
    sa, sb = set(nums_a), set(nums_b)
    common = sa & sb
    if not common:
        return 0, 0.0, 0
    frac = len(common) / max(len(sa), len(sb))
    longest = max(len(n) for n in common)
    return 1, frac, longest


def length_diff(a: str, b: str) -> int:
    return abs(len(a or '') - len(b or ''))


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a or ''), len(b or '')
    return min(la, lb) / max(la, lb) if max(la, lb) else 1.0


def token_count_diff(a: str, b: str) -> int:
    return abs(len((a or '').split()) - len((b or '').split()))


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point — compute features for a single (S1, candidate) pair
# ══════════════════════════════════════════════════════════════════════════════

def compute_pairwise_features(s1: dict, cand: dict) -> dict:
    """
    Compute all similarity features for one (S1, candidate) pair.

    Both *s1* and *cand* are dicts produced by ``normalize_record()``.
    Field keys: nl (name light), na (name aggressive), al (addr light),
    aa (addr aggressive), country, anums (address numerics).
    """
    s_nl = s1.get('nl', '')
    s_na = s1.get('na', '')
    s_al = s1.get('al', '')
    s_aa = s1.get('aa', '')
    s_c  = s1.get('country', '')
    s_an = s1.get('anums', [])

    c_nl = cand.get('nl', '')
    c_na = cand.get('na', '')
    c_al = cand.get('al', '')
    c_aa = cand.get('aa', '')
    c_c  = cand.get('country', '')
    c_an = cand.get('anums', [])

    f = {}

    # ── Name features (aggressive normalisation) ────────────────────────────
    f['name_jaccard']        = jaccard_token(s_na, c_na)
    f['name_levenshtein']    = norm_lev(s_na, c_na)
    f['name_token_sort']     = tok_sort(s_na, c_na)
    f['name_token_set']      = tok_set(s_na, c_na)
    f['name_partial']        = partial(s_na, c_na)
    f['name_wratio']         = wratio(s_na, c_na)
    f['name_monge_elkan']    = monge_elkan(s_na, c_na)
    f['name_acronym']        = acronym_match(s_na, c_na)
    f['name_exact']          = 1.0 if s_nl and s_nl == c_nl else 0.0
    f['name_exact_agg']      = 1.0 if s_na and s_na == c_na else 0.0

    # Name on light normalisation (catches over-stripped cases)
    f['name_light_levenshtein'] = norm_lev(s_nl, c_nl)
    f['name_light_token_sort']  = tok_sort(s_nl, c_nl)

    # ── Address features ────────────────────────────────────────────────────
    f['addr_jaccard']        = jaccard_token(s_aa, c_aa)
    f['addr_levenshtein']    = norm_lev(s_aa, c_aa)
    f['addr_token_sort']     = tok_sort(s_aa, c_aa)
    f['addr_token_set']      = tok_set(s_aa, c_aa)
    f['addr_partial']        = partial(s_aa, c_aa)

    nm, nf, nl_ = numeric_features(s_an, c_an)
    f['addr_numeric_match']   = nm
    f['addr_numeric_frac']    = nf
    f['addr_numeric_longest'] = nl_

    # ── Country ──────────────────────────────────────────────────────────────
    f['country_match'] = 1 if (s_c and c_c and s_c == c_c) else 0

    # ── Length / shape ───────────────────────────────────────────────────────
    f['name_length_diff']       = length_diff(s_na, c_na)
    f['name_length_ratio']      = length_ratio(s_na, c_na)
    f['addr_length_diff']       = length_diff(s_aa, c_aa)
    f['addr_length_ratio']      = length_ratio(s_aa, c_aa)
    f['name_token_count_diff']  = token_count_diff(s_na, c_na)
    f['addr_token_count_diff']  = token_count_diff(s_aa, c_aa)

    return f


# Canonical feature column lists (used by the pipeline for consistency)
FEATURE_COLUMNS = [
    'name_jaccard', 'name_levenshtein', 'name_token_sort', 'name_token_set',
    'name_partial', 'name_wratio', 'name_monge_elkan', 'name_acronym',
    'name_exact', 'name_exact_agg',
    'name_light_levenshtein', 'name_light_token_sort',
    'addr_jaccard', 'addr_levenshtein', 'addr_token_sort', 'addr_token_set',
    'addr_partial',
    'addr_numeric_match', 'addr_numeric_frac', 'addr_numeric_longest',
    'country_match',
    'name_length_diff', 'name_length_ratio',
    'addr_length_diff', 'addr_length_ratio',
    'name_token_count_diff', 'addr_token_count_diff',
]

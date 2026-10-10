"""
Candidate generation / blocking module.

Uses multiple independent blocking strategies and unions the candidates.
Targets >=99% recall with maximum reduction ratio.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from src.preprocessing.normalization import (
    get_char_ngrams, extract_name_tokens, extract_first_token
)
from src.config import BLOCKING_TOP_K, MAX_CANDIDATES_PER_S1


def build_inverted_index_first_token(df, id_col='entity_id'):
    """Build inverted index: first_token → set of entity_ids."""
    index = defaultdict(set)
    for _, row in df.iterrows():
        token = row.get('name_first_token', '')
        if token:
            index[token].add(row[id_col])
    return index


def build_inverted_index_tokens(df, id_col='entity_id'):
    """Build inverted index: each significant name token → set of entity_ids."""
    index = defaultdict(set)
    for _, row in df.iterrows():
        tokens = row.get('name_tokens', set())
        if isinstance(tokens, set):
            for t in tokens:
                index[t].add(row[id_col])
    return index


def build_inverted_index_ngrams(df, id_col='entity_id'):
    """Build inverted index: character trigram → set of entity_ids."""
    index = defaultdict(set)
    for _, row in df.iterrows():
        ngrams = row.get('name_ngrams', set())
        if isinstance(ngrams, set):
            for ng in ngrams:
                index[ng].add(row[id_col])
    return index


def build_inverted_index_addr_tokens(df, id_col='entity_id'):
    """Build inverted index for address tokens (significant ones only)."""
    index = defaultdict(set)
    stopwords = {'road', 'street', 'near', 'opposite', 'floor', 'building',
                 'number', 'block', 'plot', 'sector', 'phase', 'area',
                 'nagar', 'colony', 'village', 'town', 'city', 'state',
                 'district', 'taluka', 'tehsil', 'post', 'office',
                 'rue', 'avenue', 'boulevard', 'place', 'square',
                 'north', 'south', 'east', 'west', 'new', 'old',
                 'suite', 'apartment', 'unit', 'room', 'dept'}
    for _, row in df.iterrows():
        addr = row.get('addr_aggressive', '')
        if isinstance(addr, str) and addr:
            tokens = addr.split()
            for t in tokens:
                if t not in stopwords and len(t) > 2 and not t.isdigit():
                    index[t].add(row[id_col])
    return index


def build_inverted_index_addr_numerics(df, id_col='entity_id'):
    """Build inverted index: numeric address tokens (PIN codes, etc.) → entity_ids."""
    index = defaultdict(set)
    for _, row in df.iterrows():
        numerics = row.get('addr_numerics', [])
        if isinstance(numerics, list):
            for n in numerics:
                if len(n) >= 4:  # only meaningful numbers (PIN codes, building numbers)
                    index[n].add(row[id_col])
    return index


def block_by_first_token(s1_row, idx, country_idx=None, s1_country=None):
    """Block by matching first name token."""
    ft = s1_row.get('name_first_token', '')
    if not ft:
        return set()
    candidates = idx.get(ft, set())
    if country_idx and s1_country:
        country_candidates = country_idx.get(s1_country, set())
        filtered = candidates & country_candidates
        # If country filter removes too much, fall back to unfiltered
        if len(filtered) > 0:
            return filtered
    return candidates


def block_by_token_overlap(s1_row, idx, min_overlap=1):
    """Block by having at least min_overlap significant name tokens in common."""
    tokens = s1_row.get('name_tokens', set())
    if not isinstance(tokens, set) or not tokens:
        return set()
    candidate_counts = defaultdict(int)
    for t in tokens:
        for cid in idx.get(t, set()):
            candidate_counts[cid] += 1
    return {cid for cid, count in candidate_counts.items() if count >= min_overlap}


def block_by_ngram_overlap(s1_row, idx, min_jaccard=0.15, top_k=None):
    """
    Block by character trigram overlap (approximate Jaccard).
    Returns candidates with sufficient trigram overlap.
    """
    top_k = top_k or BLOCKING_TOP_K
    ngrams = s1_row.get('name_ngrams', set())
    if not isinstance(ngrams, set) or not ngrams:
        return set()

    candidate_overlap = defaultdict(int)
    for ng in ngrams:
        for cid in idx.get(ng, set()):
            candidate_overlap[cid] += 1

    n_query = len(ngrams)
    candidates = set()
    # Sort by overlap count descending and take top_k
    sorted_candidates = sorted(candidate_overlap.items(), key=lambda x: x[1], reverse=True)
    for cid, overlap in sorted_candidates[:top_k * 3]:
        # Approximate Jaccard: overlap / (n_query + estimated_candidate_ngrams - overlap)
        # Use a relaxed threshold to avoid missing matches
        if overlap >= max(1, int(n_query * min_jaccard)):
            candidates.add(cid)
        if len(candidates) >= top_k:
            break
    return candidates


def block_by_address_tokens(s1_row, idx, min_overlap=1):
    """Block by address token overlap."""
    addr = s1_row.get('addr_aggressive', '')
    if not isinstance(addr, str) or not addr:
        return set()
    stopwords = {'road', 'street', 'near', 'opposite', 'floor', 'building',
                 'number', 'block', 'plot', 'sector', 'phase', 'area',
                 'nagar', 'colony', 'village', 'town', 'city', 'state',
                 'district', 'taluka', 'tehsil', 'post', 'office',
                 'rue', 'avenue', 'boulevard', 'place', 'square',
                 'north', 'south', 'east', 'west', 'new', 'old',
                 'suite', 'apartment', 'unit', 'room', 'dept'}
    tokens = [t for t in addr.split() if t not in stopwords and len(t) > 2 and not t.isdigit()]
    if not tokens:
        return set()

    candidate_counts = defaultdict(int)
    for t in tokens:
        for cid in idx.get(t, set()):
            candidate_counts[cid] += 1
    return {cid for cid, count in candidate_counts.items() if count >= min_overlap}


def block_by_address_numerics(s1_row, idx):
    """Block by shared numeric tokens in address (PIN codes, building numbers)."""
    numerics = s1_row.get('addr_numerics', [])
    if not isinstance(numerics, list) or not numerics:
        return set()
    candidates = set()
    for n in numerics:
        if len(n) >= 4:
            candidates |= idx.get(n, set())
    return candidates


def generate_candidates(s1_df, s2_df, s3_df, verbose=True):
    """
    Generate candidate pairs using multiple blocking strategies, unioned.

    Returns:
        dict: {s1_entity_id: set of candidate s2/s3 entity_ids}
    """
    if verbose:
        print("Building inverted indices for S2...")
    s2_first_token_idx = build_inverted_index_first_token(s2_df)
    s2_token_idx = build_inverted_index_tokens(s2_df)
    s2_ngram_idx = build_inverted_index_ngrams(s2_df)
    s2_addr_token_idx = build_inverted_index_addr_tokens(s2_df)
    s2_addr_num_idx = build_inverted_index_addr_numerics(s2_df)

    if verbose:
        print("Building inverted indices for S3...")
    s3_first_token_idx = build_inverted_index_first_token(s3_df)
    s3_token_idx = build_inverted_index_tokens(s3_df)
    s3_ngram_idx = build_inverted_index_ngrams(s3_df)
    s3_addr_token_idx = build_inverted_index_addr_tokens(s3_df)
    s3_addr_num_idx = build_inverted_index_addr_numerics(s3_df)

    # Optional: country index for soft filtering
    s2_country_idx = defaultdict(set)
    s3_country_idx = defaultdict(set)
    if 'country_normalized' in s2_df.columns:
        for _, row in s2_df.iterrows():
            c = row.get('country_normalized', '')
            if c:
                s2_country_idx[c].add(row['entity_id'])
    if 'country_normalized' in s3_df.columns:
        for _, row in s3_df.iterrows():
            c = row.get('country_normalized', '')
            if c:
                s3_country_idx[c].add(row['entity_id'])

    candidates = {}
    iterator = tqdm(s1_df.iterrows(), total=len(s1_df), desc="Blocking") if verbose else s1_df.iterrows()

    for _, s1_row in iterator:
        s1_id = s1_row['entity_id']
        s1_country = s1_row.get('country_normalized', '')
        cands = set()

        # Strategy 1: First token blocking
        cands |= block_by_first_token(s1_row, s2_first_token_idx, s2_country_idx, s1_country)
        cands |= block_by_first_token(s1_row, s3_first_token_idx, s3_country_idx, s1_country)

        # Strategy 2: Token-set blocking (any shared significant token)
        cands |= block_by_token_overlap(s1_row, s2_token_idx, min_overlap=1)
        cands |= block_by_token_overlap(s1_row, s3_token_idx, min_overlap=1)

        # Strategy 3: Character n-gram blocking (catches typos, reordering)
        cands |= block_by_ngram_overlap(s1_row, s2_ngram_idx, min_jaccard=0.15)
        cands |= block_by_ngram_overlap(s1_row, s3_ngram_idx, min_jaccard=0.15)

        # Strategy 4: Address token overlap (secondary)
        addr_cands_s2 = block_by_address_tokens(s1_row, s2_addr_token_idx, min_overlap=2)
        addr_cands_s3 = block_by_address_tokens(s1_row, s3_addr_token_idx, min_overlap=2)
        cands |= addr_cands_s2
        cands |= addr_cands_s3

        # Strategy 5: Address numeric matching (PIN codes)
        cands |= block_by_address_numerics(s1_row, s2_addr_num_idx)
        cands |= block_by_address_numerics(s1_row, s3_addr_num_idx)

        # Cap candidates per S1 entity to control computation
        if len(cands) > MAX_CANDIDATES_PER_S1:
            # Prioritize by strategy overlap - keep those found by multiple strategies
            # For now, just cap to avoid explosion
            cands_list = list(cands)
            cands = set(cands_list[:MAX_CANDIDATES_PER_S1])

        candidates[s1_id] = cands

    return candidates


def measure_blocking_recall(candidates, ground_truth):
    """
    Measure what fraction of true matches survive blocking.
    This is the recall ceiling.
    """
    total_true_matches = 0
    found_true_matches = 0
    missed = []

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands = candidates.get(s1_id, set())
        for match_id in true_matches:
            total_true_matches += 1
            if match_id in cands:
                found_true_matches += 1
            else:
                missed.append((s1_id, match_id))

    recall = found_true_matches / total_true_matches if total_true_matches > 0 else 1.0

    # Reduction ratio
    total_possible = 0
    total_candidates = 0
    n_s2_s3 = 0  # we don't have this here, compute from candidates
    for s1_id, cands in candidates.items():
        total_candidates += len(cands)

    return {
        'recall': recall,
        'total_true_matches': total_true_matches,
        'found_true_matches': found_true_matches,
        'missed_count': len(missed),
        'missed_examples': missed[:20],
        'total_candidates': total_candidates,
        'avg_candidates_per_s1': total_candidates / len(candidates) if candidates else 0,
    }

"""
Threshold optimization module.

Implements rank-aware / gap-based thresholding optimized for F0.5.
"""
import numpy as np
from src.evaluation.metrics import macro_f05


def apply_threshold(pair_scores, threshold):
    """
    Apply a simple global threshold.

    Args:
        pair_scores: list of (s1_id, candidate_id, probability)
        threshold: probability threshold

    Returns:
        dict {s1_id: set of matched candidate ids}
    """
    predictions = {}
    for s1_id, cand_id, prob in pair_scores:
        if s1_id not in predictions:
            predictions[s1_id] = set()
        if prob >= threshold:
            predictions[s1_id].add(cand_id)
    return predictions


def apply_rank_aware_threshold(pair_scores, tau1=0.5, gap_threshold=0.15, max_matches=10):
    """
    Rank-aware / gap-based thresholding:
    1. Require top-1 candidate prob >= tau1 to accept any match.
    2. Accept additional matches only if their prob >= tau1 AND
       the gap to the next candidate is not too large (indicating clear separation).

    Args:
        pair_scores: list of (s1_id, candidate_id, probability)
        tau1: minimum probability for top-1 candidate
        gap_threshold: minimum absolute gap from current to next candidate
                       to stop accepting more matches
        max_matches: hard cap on matches per entity

    Returns:
        dict {s1_id: set of matched candidate ids}
    """
    # Group by S1 entity and sort by probability descending
    from collections import defaultdict
    grouped = defaultdict(list)
    for s1_id, cand_id, prob in pair_scores:
        grouped[s1_id].append((cand_id, prob))

    predictions = {}
    for s1_id, cands in grouped.items():
        # Sort by probability descending
        cands.sort(key=lambda x: x[1], reverse=True)
        matched = set()

        if not cands or cands[0][1] < tau1:
            # Top-1 doesn't meet threshold → singleton
            predictions[s1_id] = set()
            continue

        # Accept top-1
        matched.add(cands[0][0])

        # Consider additional matches
        for i in range(1, min(len(cands), max_matches)):
            cand_id, prob = cands[i]
            if prob < tau1:
                break

            # Check gap: if there's a big drop from previous, stop
            prev_prob = cands[i - 1][1]
            gap = prev_prob - prob
            if gap > gap_threshold:
                break

            matched.add(cand_id)

        predictions[s1_id] = matched

    return predictions


def optimize_threshold_simple(pair_scores, ground_truth, thresholds=None):
    """
    Sweep a global threshold to maximize F0.5 on validation data.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.96, 0.02)

    best_f05 = -1
    best_threshold = 0.5
    results = []

    for t in thresholds:
        preds = apply_threshold(pair_scores, t)
        # Ensure all GT entities are present
        for s1_id in ground_truth:
            if s1_id not in preds:
                preds[s1_id] = set()
        metrics = macro_f05(preds, ground_truth)
        results.append((t, metrics['f05_macro'], metrics['precision_macro'], metrics['recall_macro']))
        if metrics['f05_macro'] > best_f05:
            best_f05 = metrics['f05_macro']
            best_threshold = t

    print(f"\n{'Threshold':>10s} {'F0.5':>8s} {'Precision':>10s} {'Recall':>8s}")
    print("-" * 40)
    for t, f05, prec, rec in results:
        marker = " <- BEST" if t == best_threshold else ""
        print(f"{t:10.3f} {f05:8.4f} {prec:10.4f} {rec:8.4f}{marker}")

    return best_threshold, best_f05


def optimize_rank_aware_threshold(pair_scores, ground_truth,
                                  tau1_range=None, gap_range=None):
    """
    Grid search over rank-aware threshold parameters to maximize F0.5.
    """
    if tau1_range is None:
        tau1_range = np.arange(0.2, 0.91, 0.05)
    if gap_range is None:
        gap_range = np.arange(0.05, 0.51, 0.05)

    best_f05 = -1
    best_params = {'tau1': 0.5, 'gap': 0.15}
    all_results = []

    for tau1 in tau1_range:
        for gap in gap_range:
            preds = apply_rank_aware_threshold(pair_scores, tau1=tau1, gap_threshold=gap)
            for s1_id in ground_truth:
                if s1_id not in preds:
                    preds[s1_id] = set()
            metrics = macro_f05(preds, ground_truth)
            all_results.append((tau1, gap, metrics))
            if metrics['f05_macro'] > best_f05:
                best_f05 = metrics['f05_macro']
                best_params = {'tau1': tau1, 'gap': gap}

    print(f"\nBest rank-aware threshold: tau1={best_params['tau1']:.3f}, gap={best_params['gap']:.3f}")
    print(f"Best F0.5: {best_f05:.4f}")

    # Show top 10 parameter combos
    all_results.sort(key=lambda x: x[2]['f05_macro'], reverse=True)
    print(f"\n{'tau1':>6s} {'gap':>6s} {'F0.5':>8s} {'Prec':>8s} {'Rec':>8s} {'Sing-F05':>9s} {'Multi-F05':>10s}")
    print("-" * 60)
    for tau1, gap, m in all_results[:10]:
        print(f"{tau1:6.3f} {gap:6.3f} {m['f05_macro']:8.4f} {m['precision_macro']:8.4f} "
              f"{m['recall_macro']:8.4f} {m['singleton_f05']:9.4f} {m['multi_match_f05']:10.4f}")

    return best_params, best_f05

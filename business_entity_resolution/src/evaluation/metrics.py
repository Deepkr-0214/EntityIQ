"""
F0.5 evaluation metrics module.

Macro-averaged per S1 entity (singletons included).
"""
import numpy as np


def f_beta_score(precision, recall, beta=0.5):
    """Compute F-beta score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def entity_f05(predicted_set, true_set):
    """
    Compute F0.5 for a single entity.

    - Singleton with correct empty prediction: 1.0
    - Singleton with any false prediction: 0.0
    - Non-singleton: standard F0.5
    """
    # Both empty: correct singleton identification
    if not true_set and not predicted_set:
        return 1.0, 1.0, 1.0  # f05, precision, recall

    # True is empty but we predicted something: false merges
    if not true_set and predicted_set:
        return 0.0, 0.0, 1.0

    # True is non-empty but we predicted nothing: missed everything
    if true_set and not predicted_set:
        return 0.0, 1.0, 0.0  # precision is undefined, but no false merges

    # Both non-empty
    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = f_beta_score(precision, recall, beta=0.5)

    return f05, precision, recall


def macro_f05(predictions, ground_truth):
    """
    Macro-averaged F0.5 over all S1 entities.

    Args:
        predictions: dict {s1_id: set of predicted matched ids}
        ground_truth: dict {s1_id: set of true matched ids}

    Returns:
        dict with overall metrics and per-category breakdowns
    """
    f05_scores = []
    precision_scores = []
    recall_scores = []

    singleton_scores = []
    single_match_scores = []
    multi_match_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05, prec, rec = entity_f05(pred_set, true_set)
        f05_scores.append(f05)
        precision_scores.append(prec)
        recall_scores.append(rec)

        # Category breakdown
        if not true_set:
            singleton_scores.append(f05)
        elif len(true_set) == 1:
            single_match_scores.append(f05)
        else:
            multi_match_scores.append(f05)

    return {
        'f05_macro': np.mean(f05_scores) if f05_scores else 0.0,
        'precision_macro': np.mean(precision_scores) if precision_scores else 0.0,
        'recall_macro': np.mean(recall_scores) if recall_scores else 0.0,
        'n_entities': len(f05_scores),
        # Category breakdowns
        'singleton_f05': np.mean(singleton_scores) if singleton_scores else 0.0,
        'singleton_count': len(singleton_scores),
        'single_match_f05': np.mean(single_match_scores) if single_match_scores else 0.0,
        'single_match_count': len(single_match_scores),
        'multi_match_f05': np.mean(multi_match_scores) if multi_match_scores else 0.0,
        'multi_match_count': len(multi_match_scores),
    }

"""
utils.py

Utility functions for Multiple Choice Question Answering
"""

import numpy as np


# ----------------------------
# Label Mappings
# ----------------------------

LABEL2ID = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

ID2LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}


# ----------------------------
# Build Input Text
# ----------------------------

def build_text(row):
    """
    Convert one dataframe row into the input format
    expected by the transformer models.
    """

    return (
        f"Question: {row['prompt']}"
        f"\nA. {row['A']}"
        f"\nB. {row['B']}"
        f"\nC. {row['C']}"
        f"\nD. {row['D']}"
        f"\nE. {row['E']}"
    )


# ----------------------------
# Soft Voting Ensemble
# ----------------------------

def weighted_ensemble(
    deberta_probs,
    roberta_probs,
    deberta_weight=0.7,
    roberta_weight=0.3,
):
    """
    Weighted probability averaging.
    """

    return (
        deberta_weight * deberta_probs
        + roberta_weight * roberta_probs
    )


# ----------------------------
# Top-k Prediction
# ----------------------------

def get_topk_labels(probabilities, k=3):
    """
    Returns labels ordered by probability.

    Example:
    ['A','C','B']
    """

    topk = np.argsort(probabilities)[::-1][:k]

    return [ID2LABEL[i] for i in topk]


def prediction_string(probabilities, k=3):
    """
    Returns Kaggle submission format.

    Example:
    A C B
    """

    return " ".join(get_topk_labels(probabilities, k))


# ----------------------------
# MAP@3
# ----------------------------

def apk(actual, predicted, k=3):
    """
    Average Precision at K
    """

    predicted = predicted[:k]

    for i, p in enumerate(predicted):

        if p == actual:
            return 1.0 / (i + 1)

    return 0.0


def mapk(actuals, predictions, k=3):
    """
    Mean Average Precision at K
    """

    scores = [
        apk(a, p, k)
        for a, p in zip(actuals, predictions)
    ]

    return np.mean(scores)


# ----------------------------
# Confidence
# ----------------------------

def confidence(probabilities):
    """
    Highest class probability.
    """

    return float(np.max(probabilities))


def confidence_gain(
    deberta_probs,
    ensemble_probs,
):
    """
    Confidence gain after ensembling.
    """

    return (
        confidence(ensemble_probs)
        - confidence(deberta_probs)
    )
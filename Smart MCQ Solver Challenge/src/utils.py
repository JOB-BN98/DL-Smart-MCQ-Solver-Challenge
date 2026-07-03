"""
utils.py

Utility functions for the Smart MCQ Solver
Milestone 2

Contains helper functions for:
- Loading the transformer model
- Generating embeddings
- Computing cosine similarity
- Ranking answer options
- mAP@3 evaluation
"""

import pandas as pd

from sentence_transformers import SentenceTransformer, util


OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


# -----------------------------------------------------
# Load Sentence Transformer
# -----------------------------------------------------

def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the SentenceTransformer model.
    """

    model = SentenceTransformer(model_name)
    return model


# -----------------------------------------------------
# Read CSV
# -----------------------------------------------------

def load_dataset(csv_path):
    """
    Load a CSV dataset.
    """

    return pd.read_csv(csv_path)


# -----------------------------------------------------
# Generate Embedding
# -----------------------------------------------------

def get_embedding(model, text):
    """
    Generate contextual embedding for text.
    """

    return model.encode(
        text,
        convert_to_tensor=True
    )


# -----------------------------------------------------
# Cosine Similarity
# -----------------------------------------------------

def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity.
    """

    return util.cos_sim(
        embedding1,
        embedding2
    ).item()


# -----------------------------------------------------
# Rank Options
# -----------------------------------------------------

def rank_options(model, prompt, options):
    """
    Rank answer options according to cosine similarity.
    """

    prompt_embedding = get_embedding(
        model,
        prompt
    )

    scores = {}

    for option_name, option_text in options.items():

        option_embedding = get_embedding(
            model,
            option_text
        )

        scores[option_name] = cosine_similarity(
            prompt_embedding,
            option_embedding
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# -----------------------------------------------------
# Top 3 Prediction
# -----------------------------------------------------

def predict_top3(model, row):
    """
    Return Top-3 predicted options.
    """

    options = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"],
        "E": row["E"],
    }

    ranked = rank_options(
        model,
        row["prompt"],
        options
    )

    return [option for option, _ in ranked[:3]]


# -----------------------------------------------------
# mAP@3
# -----------------------------------------------------

def map_at_3(actual, predicted):
    """
    Average Precision@3 for one sample.
    """

    for i, prediction in enumerate(predicted[:3]):

        if prediction == actual:
            return 1.0 / (i + 1)

    return 0.0


# -----------------------------------------------------
# Dataset mAP@3
# -----------------------------------------------------

def evaluate_map3(actual_answers, predictions):
    """
    Compute dataset mAP@3.
    """

    score = 0

    for actual, predicted in zip(actual_answers, predictions):
        score += map_at_3(actual, predicted)

    return score / len(actual_answers)


# -----------------------------------------------------
# Create Submission
# -----------------------------------------------------

def create_submission(ids, predictions):
    """
    Create submission DataFrame.
    """

    prediction_strings = [
        " ".join(pred)
        for pred in predictions
    ]

    return pd.DataFrame({
        "id": ids,
        "prediction": prediction_strings
    })
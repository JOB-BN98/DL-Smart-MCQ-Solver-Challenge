"""
utils.py
---------

Utility functions for the Smart MCQ Solver.

Contains:
- Dataset loading
- Text preprocessing
- Corpus creation
- Model save/load
- Prediction utilities
"""

import os
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

OPTIONS = ["A", "B", "C", "D", "E"]


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

def load_dataset(path):
    """
    Load CSV dataset.
    """
    return pd.read_csv(path)


# -------------------------------------------------------
# Text Processing
# -------------------------------------------------------

def clean_text(text):
    """
    Basic preprocessing.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = " ".join(text.split())

    return text.lower()


def prepare_corpus(df):
    """
    Create corpus used for TF-IDF training.
    """

    corpus = (
        df["prompt"].fillna("").apply(clean_text) + " " +
        df["A"].fillna("").apply(clean_text) + " " +
        df["B"].fillna("").apply(clean_text) + " " +
        df["C"].fillna("").apply(clean_text) + " " +
        df["D"].fillna("").apply(clean_text) + " " +
        df["E"].fillna("").apply(clean_text)
    )

    return corpus.tolist()


# -------------------------------------------------------
# Model Save / Load
# -------------------------------------------------------

def save_model(model, output_path):
    """
    Save trained model.
    """

    directory = os.path.dirname(output_path)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    Load saved model.
    """

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

def predict_top3(row, vectorizer):
    """
    Predict Top-3 answer choices.
    """

    prompt_vector = vectorizer.transform(
        [clean_text(row["prompt"])]
    )

    scores = {}

    for option in OPTIONS:

        option_vector = vectorizer.transform(
            [clean_text(row[option])]
        )

        similarity = cosine_similarity(
            prompt_vector,
            option_vector
        )[0][0]

        scores[option] = similarity

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return ranked[:3]


def generate_predictions(df, vectorizer):
    """
    Generate predictions for entire dataframe.
    """

    predictions = []

    for _, row in df.iterrows():

        top3 = predict_top3(
            row,
            vectorizer
        )

        predictions.append(" ".join(top3))

    return predictions


# -------------------------------------------------------
# Submission
# -------------------------------------------------------

def create_submission(df, predictions):
    """
    Create Kaggle submission dataframe.
    """

    submission = pd.DataFrame({
        "id": df["id"],
        "prediction": predictions
    })

    return submission


# -------------------------------------------------------
# Statistics
# -------------------------------------------------------

def print_dataset_info(df):

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    print(f"Samples : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns")

    for column in df.columns:
        print(f" - {column}")

    print()


def print_model_info(vectorizer):

    print("=" * 50)
    print("Model Information")
    print("=" * 50)

    print(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")

    print()
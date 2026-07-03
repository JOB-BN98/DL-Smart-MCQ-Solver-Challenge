"""
inference.py
-------------
Inference script for the Smart MCQ Solver (Milestone 1)

Usage:
    python inference.py --train train.csv --test test.csv

Output:
    submission.csv
"""

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OPTIONS = ["A", "B", "C", "D", "E"]


def build_vectorizer(train_df):
    """
    Fit TF-IDF on all available text in the training set.
    """

    combined_text = (
        train_df["prompt"].fillna("") + " " +
        train_df["A"].fillna("") + " " +
        train_df["B"].fillna("") + " " +
        train_df["C"].fillna("") + " " +
        train_df["D"].fillna("") + " " +
        train_df["E"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectorizer.fit(combined_text)

    return vectorizer


def predict_row(row, vectorizer):
    """
    Predict Top-3 answer choices for one MCQ.
    """

    prompt_vec = vectorizer.transform(
        [str(row["prompt"])]
    )

    scores = {}

    for option in OPTIONS:

        option_vec = vectorizer.transform(
            [str(row[option])]
        )

        similarity = cosine_similarity(
            prompt_vec,
            option_vec
        )[0][0]

        scores[option] = similarity

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return ranked[:3]


def generate_submission(test_df, vectorizer):

    predictions = []

    for _, row in test_df.iterrows():

        top3 = predict_row(
            row,
            vectorizer
        )

        predictions.append(
            " ".join(top3)
        )

    submission = pd.DataFrame({
        "id": test_df["id"],
        "prediction": predictions
    })

    return submission


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train.csv"
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test.csv"
    )

    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV filename"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Loading datasets")
    print("=" * 60)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    print(f"Training samples : {len(train_df)}")
    print(f"Testing samples  : {len(test_df)}")

    print("\nBuilding TF-IDF Vectorizer...")
    vectorizer = build_vectorizer(train_df)

    print("Running inference...")
    submission = generate_submission(
        test_df,
        vectorizer
    )

    submission.to_csv(
        args.output,
        index=False
    )

    print(f"\nSaved predictions to {args.output}")

    print("\nSample Predictions")
    print(submission.head())


if __name__ == "__main__":
    main()
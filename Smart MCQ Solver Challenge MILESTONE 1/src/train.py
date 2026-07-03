"""
train.py
---------
Train the TF-IDF model for the Smart MCQ Solver.

Usage:
    python train.py --train train.csv

Outputs:
    models/tfidf_vectorizer.pkl
"""

import os
import argparse
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(path):
    """Load training dataset."""
    return pd.read_csv(path)


def prepare_corpus(df):
    """
    Build corpus from prompt and answer options.
    """

    corpus = (
        df["prompt"].fillna("") + " " +
        df["A"].fillna("") + " " +
        df["B"].fillna("") + " " +
        df["C"].fillna("") + " " +
        df["D"].fillna("") + " " +
        df["E"].fillna("")
    )

    return corpus.tolist()


def train_vectorizer(corpus):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2)
    )

    vectorizer.fit(corpus)

    return vectorizer


def save_model(model, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        "tfidf_vectorizer.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to:\n{save_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train.csv"
    )

    parser.add_argument(
        "--output_dir",
        default="models",
        help="Directory to save trained model"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    df = load_dataset(args.train)

    print(f"Samples : {len(df)}")

    print("\nPreparing corpus...")
    corpus = prepare_corpus(df)

    print("Training TF-IDF Vectorizer...")
    vectorizer = train_vectorizer(corpus)

    print(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")

    save_model(
        vectorizer,
        args.output_dir
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
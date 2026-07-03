"""
train.py

Milestone 2
Transformer-based Semantic Matching

This script does NOT fine-tune a transformer.

Instead it

1. Loads train.csv
2. Generates contextual embeddings
3. Computes cosine similarity
4. Evaluates mAP@3
5. Saves embeddings for future use
"""

import argparse
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util

OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded.\n")


# ---------------------------------------------------
# mAP@3
# ---------------------------------------------------

def map_at_3(actual, predicted):

    score = 0.0

    for i, pred in enumerate(predicted[:3]):

        if pred == actual:
            score = 1.0 / (i + 1)
            break

    return score


# ---------------------------------------------------
# Train (Embedding Generation)
# ---------------------------------------------------

def train(train_csv):

    df = pd.read_csv(train_csv)

    predictions = []

    prompt_embeddings = []

    option_embeddings = []

    total_score = 0

    print("Generating embeddings...\n")

    for idx, row in df.iterrows():

        prompt_emb = model.encode(
            row["prompt"],
            convert_to_tensor=True
        )

        prompt_embeddings.append(
            prompt_emb.cpu().numpy()
        )

        similarities = {}

        current_option_embeddings = {}

        for option in OPTION_COLUMNS:

            emb = model.encode(
                row[option],
                convert_to_tensor=True
            )

            current_option_embeddings[option] = emb.cpu().numpy()

            similarities[option] = util.cos_sim(
                prompt_emb,
                emb
            ).item()

        option_embeddings.append(current_option_embeddings)

        ranked = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top3 = [x[0] for x in ranked[:3]]

        predictions.append(" ".join(top3))

        total_score += map_at_3(
            row["answer"],
            top3
        )

    df["prediction"] = predictions

    map3 = total_score / len(df)

    print("=" * 40)
    print(f"Training mAP@3 : {map3:.4f}")
    print("=" * 40)

    np.save(
        "prompt_embeddings.npy",
        np.array(prompt_embeddings)
    )

    np.save(
        "option_embeddings.npy",
        np.array(option_embeddings, dtype=object)
    )

    df.to_csv(
        "train_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print(" prompt_embeddings.npy")
    print(" option_embeddings.npy")
    print(" train_predictions.csv")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train.csv"
    )

    args = parser.parse_args()

    train(args.train)
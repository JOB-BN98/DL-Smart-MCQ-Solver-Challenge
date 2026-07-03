"""
inference.py

Inference pipeline for the Smart MCQ Solver
Milestone 2

Method:
- Load SentenceTransformer (all-MiniLM-L6-v2)
- Generate embeddings
- Compute cosine similarity
- Rank answer options
- Produce Top-3 predictions
"""

import argparse
import pandas as pd

from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("Loading Sentence Transformer...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model Loaded.\n")


OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------
# Predict Top-3
# ---------------------------------------------------

def predict_top3(row):

    prompt_embedding = model.encode(
        row["prompt"],
        convert_to_tensor=True
    )

    scores = {}

    for option in OPTION_COLUMNS:

        option_embedding = model.encode(
            row[option],
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            prompt_embedding,
            option_embedding
        ).item()

        scores[option] = similarity

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top3 = [x[0] for x in ranked[:3]]

    return " ".join(top3)


# ---------------------------------------------------
# Run Inference
# ---------------------------------------------------

def run_inference(test_csv, output_csv):

    print(f"Reading {test_csv}")

    df = pd.read_csv(test_csv)

    print("Generating predictions...")

    df["prediction"] = df.apply(
        predict_top3,
        axis=1
    )

    submission = pd.DataFrame({
        "id": df["id"],
        "prediction": df["prediction"]
    })

    submission.to_csv(
        output_csv,
        index=False
    )

    print(f"\nSubmission saved to {output_csv}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV"
    )

    args = parser.parse_args()

    run_inference(
        args.test,
        args.output
    )
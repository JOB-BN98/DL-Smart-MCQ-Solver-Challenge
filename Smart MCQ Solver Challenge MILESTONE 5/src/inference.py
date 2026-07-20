import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBERTA_PATH = "deberta_checkpoint"
ROBERTA_PATH = "roberta_checkpoint"

TEST_FILE = "test.csv"
OUTPUT_FILE = "submission.csv"

DEBERTA_WEIGHT = 0.7
ROBERTA_WEIGHT = 0.3

ID2LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}


def build_text(row):
    return (
        f"Question: {row['prompt']}"
        f"\nA. {row['A']}"
        f"\nB. {row['B']}"
        f"\nC. {row['C']}"
        f"\nD. {row['D']}"
        f"\nE. {row['E']}"
    )


def predict_probabilities(model, tokenizer, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def main():

    print("Loading models...")

    deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_PATH
    ).to(DEVICE)

    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_PATH
    ).to(DEVICE)

    deberta_model.eval()
    roberta_model.eval()

    print("Reading test file...")

    test_df = pd.read_csv(TEST_FILE)

    predictions = []

    print("Running inference...")

    for _, row in test_df.iterrows():

        text = build_text(row)

        deberta_probs = predict_probabilities(
            deberta_model,
            deberta_tokenizer,
            text
        )

        roberta_probs = predict_probabilities(
            roberta_model,
            roberta_tokenizer,
            text
        )

        final_probs = (
            DEBERTA_WEIGHT * deberta_probs +
            ROBERTA_WEIGHT * roberta_probs
        )

        top3 = np.argsort(final_probs)[::-1][:3]

        prediction = " ".join(
            ID2LABEL[i] for i in top3
        )

        predictions.append(prediction)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "prediction": predictions
    })

    submission.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDone!")
    print(submission.head())
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
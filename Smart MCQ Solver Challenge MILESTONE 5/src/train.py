import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# ---------------- CONFIG ----------------

TRAIN_FILE = "train.csv"

MODELS = {
    "deberta": "microsoft/deberta-v3-small",
    "roberta": "roberta-base"
}

MAX_LENGTH = 256
NUM_LABELS = 5

label2id = {"A":0,"B":1,"C":2,"D":3,"E":4}
id2label = {v:k for k,v in label2id.items()}

# ---------------- LOAD DATA ----------------

df = pd.read_csv(TRAIN_FILE)

df["label"] = df["answer"].map(label2id)

train_data, valid_data = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)

# ---------------- METRICS ----------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ---------------- TRAIN FUNCTION ----------------

def train_model(model_name, checkpoint_name):

    print(f"\nTraining {checkpoint_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):

        text = (
            "Question: " + example["prompt"] +
            "\nA. " + example["A"] +
            "\nB. " + example["B"] +
            "\nC. " + example["C"] +
            "\nD. " + example["D"] +
            "\nE. " + example["E"]
        )

        return tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_ds = train_dataset.map(preprocess)
    valid_ds = valid_dataset.map(preprocess)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(

        output_dir=f"./{checkpoint_name}",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        fp16=True,

        load_best_model_at_end=True,

        logging_steps=50,

        report_to="none"
    )

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_ds,

        eval_dataset=valid_ds,

        tokenizer=tokenizer,

        data_collator=DataCollatorWithPadding(tokenizer),

        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(f"{checkpoint_name}_checkpoint")

    tokenizer.save_pretrained(f"{checkpoint_name}_checkpoint")

    print(f"Saved -> {checkpoint_name}_checkpoint")

# ---------------- MAIN ----------------

if __name__ == "__main__":

    train_model(
        MODELS["deberta"],
        "deberta"
    )

    train_model(
        MODELS["roberta"],
        "roberta"
    )

    print("\nTraining Complete!")
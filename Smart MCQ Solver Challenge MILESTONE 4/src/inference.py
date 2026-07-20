"""
inference.py

Milestone 3 - Context Augmentation with RAG Pipelines

Pipeline:
1. Build Knowledge Base
2. Embed documents using all-MiniLM-L6-v2
3. Create FAISS Index
4. Retrieve Top-K documents
5. Rerank using Cross-Encoder
6. Augment prompt with best document
7. Predict using Zero-Shot Classification
"""

import argparse
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline


class RAGPipeline:

    def __init__(self, train_csv, top_k=5):

        self.top_k = top_k

        print("Loading dataset...")
        self.train = pd.read_csv(train_csv)

        print("Building Knowledge Base...")
        self.kb = []

        for _, row in self.train.iterrows():
            correct_letter = row["answer"]
            self.kb.append(str(row[correct_letter]))

        print("Loading Embedding Model...")
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Generating Embeddings...")
        self.kb_embeddings = self.embedding_model.encode(
            self.kb,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print("Creating FAISS Index...")
        self.index = faiss.IndexFlatL2(
            self.kb_embeddings.shape[1]
        )

        self.index.add(self.kb_embeddings)

        print("Loading Cross Encoder...")
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Loading Zero-Shot Classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        print("Pipeline Ready.\n")

    def retrieve(self, prompt):

        embedding = self.embedding_model.encode(
            [prompt],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            embedding,
            self.top_k
        )

        docs = [self.kb[i] for i in indices[0]]

        return docs

    def rerank(self, prompt, docs):

        pairs = [[prompt, doc] for doc in docs]

        scores = self.cross_encoder.predict(pairs)

        best_index = int(np.argmax(scores))

        return docs[best_index]

    def predict(self, prompt, options):

        docs = self.retrieve(prompt)

        best_doc = self.rerank(prompt, docs)

        rag_prompt = (
            f"Context: {best_doc} "
            f"Question: {prompt}"
        )

        result = self.classifier(
            rag_prompt,
            candidate_labels=options,
            multi_label=False
        )

        return result

    def predict_letters(self, prompt, option_dict):

        option_texts = list(option_dict.values())

        result = self.predict(
            prompt,
            option_texts
        )

        reverse_map = {
            v: k
            for k, v in option_dict.items()
        }

        ranked_letters = [
            reverse_map[label]
            for label in result["labels"]
        ]

        return ranked_letters, result["scores"]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv",
        default="train.csv"
    )

    parser.add_argument(
        "--row",
        type=int,
        default=0
    )

    args = parser.parse_args()

    rag = RAGPipeline(args.train_csv)

    row = rag.train.iloc[args.row]

    prompt = str(row["prompt"])

    options = {
        "A": str(row["A"]),
        "B": str(row["B"]),
        "C": str(row["C"]),
        "D": str(row["D"]),
        "E": str(row["E"])
    }

    ranked_letters, scores = rag.predict_letters(
        prompt,
        options
    )

    print("=" * 60)
    print("Prompt")
    print("=" * 60)
    print(prompt)

    print("\nOptions")

    for letter, text in options.items():
        print(f"{letter}: {text}")

    print("\nPrediction Ranking")

    for letter, score in zip(ranked_letters, scores):
        print(f"{letter}: {score:.4f}")

    print("\nTop Prediction:", ranked_letters[0])

    print("Ground Truth:", row["answer"])


if __name__ == "__main__":
    main()
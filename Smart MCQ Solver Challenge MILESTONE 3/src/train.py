"""
train.py

Milestone 3 - Context Augmentation with RAG Pipelines

This script builds the Retrieval-Augmented Generation (RAG)
knowledge base by:

1. Loading train.csv
2. Extracting the correct answers
3. Creating sentence embeddings
4. Building a FAISS vector index
5. Saving the index and knowledge base for inference
"""

import os
import pickle

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


DATA_PATH = "train.csv"
OUTPUT_DIR = "artifacts"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_knowledge_base(df):
    """Extract the correct answer text from each row."""

    kb = []

    for _, row in df.iterrows():
        answer_letter = row["answer"]
        kb.append(str(row[answer_letter]))

    return kb


def create_embeddings(model, kb):
    """Generate dense embeddings."""

    embeddings = model.encode(
        kb,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)


def build_faiss_index(embeddings):
    """Create FAISS index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def save_artifacts(index, kb):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    faiss.write_index(
        index,
        os.path.join(OUTPUT_DIR, "faiss.index")
    )

    with open(
        os.path.join(OUTPUT_DIR, "knowledge_base.pkl"),
        "wb"
    ) as f:
        pickle.dump(kb, f)


def main():

    print("=" * 60)
    print("Milestone 3 - RAG Knowledge Base Builder")
    print("=" * 60)

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Size : {len(df)}")

    print("\nBuilding Knowledge Base...")
    kb = build_knowledge_base(df)

    print(f"Knowledge Base Documents : {len(kb)}")

    print("\nLoading Embedding Model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nGenerating Embeddings...")
    embeddings = create_embeddings(model, kb)

    print(f"Embedding Shape : {embeddings.shape}")

    print("\nCreating FAISS Index...")
    index = build_faiss_index(embeddings)

    print(f"Documents Indexed : {index.ntotal}")

    print("\nSaving Artifacts...")
    save_artifacts(index, kb)

    print("\nDone!")

    print("\nSaved Files")
    print("----------------------------")
    print("artifacts/")
    print(" ├── faiss.index")
    print(" └── knowledge_base.pkl")


if __name__ == "__main__":
    main()
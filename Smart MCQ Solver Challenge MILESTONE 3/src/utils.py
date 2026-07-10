"""
utils.py

Utility functions for Milestone 3
Context Augmentation with RAG Pipelines
"""

import os
import pickle

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline


# =====================================================
# Model Loading
# =====================================================

def load_embedding_model():
    """Load the Sentence Transformer embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_cross_encoder():
    """Load the Cross-Encoder reranker."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_zero_shot_classifier():
    """Load the zero-shot classifier."""
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )


# =====================================================
# Dataset
# =====================================================

def load_dataset(csv_path):
    """Load train.csv."""
    return pd.read_csv(csv_path)


# =====================================================
# Knowledge Base
# =====================================================

def build_knowledge_base(df):
    """
    Build the Knowledge Base using
    the correct option text.
    """

    kb = []

    for _, row in df.iterrows():
        answer_letter = row["answer"]
        kb.append(str(row[answer_letter]))

    return kb


# =====================================================
# Embeddings
# =====================================================

def create_embeddings(model, texts):
    """
    Generate sentence embeddings.
    """

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)


# =====================================================
# FAISS
# =====================================================

def build_faiss_index(embeddings):
    """
    Create a FAISS L2 index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def save_faiss_index(index, path):
    """
    Save FAISS index.
    """

    faiss.write_index(index, path)


def load_faiss_index(path):
    """
    Load FAISS index.
    """

    return faiss.read_index(path)


# =====================================================
# Knowledge Base Persistence
# =====================================================

def save_knowledge_base(kb, path):
    """
    Save Knowledge Base.
    """

    with open(path, "wb") as f:
        pickle.dump(kb, f)


def load_knowledge_base(path):
    """
    Load Knowledge Base.
    """

    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================
# Retrieval
# =====================================================

def retrieve_documents(
    prompt,
    embedding_model,
    index,
    kb,
    top_k=5
):
    """
    Retrieve Top-K documents.
    """

    query_embedding = embedding_model.encode(
        [prompt],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    docs = [kb[i] for i in indices[0]]

    return docs, indices[0], distances[0]


# =====================================================
# Reranking
# =====================================================

def rerank_documents(
    prompt,
    documents,
    cross_encoder
):
    """
    Rerank retrieved documents.
    """

    pairs = [
        [prompt, doc]
        for doc in documents
    ]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =====================================================
# RAG Prompt
# =====================================================

def build_rag_prompt(
    prompt,
    context
):
    """
    Construct the RAG prompt.
    """

    return f"Context: {context} Question: {prompt}"


# =====================================================
# Zero-Shot Prediction
# =====================================================

def zero_shot_predict(
    classifier,
    prompt,
    candidate_labels
):
    """
    Run zero-shot classification.
    """

    return classifier(
        prompt,
        candidate_labels=candidate_labels,
        multi_label=False
    )


# =====================================================
# Metrics
# =====================================================

def hit_rate(hits, total):
    """
    Compute Hit Rate.
    """

    return (hits / total) * 100.0


def map_at_3(
    predictions,
    true_answer
):
    """
    Compute MAP@3 for one example.

    predictions:
        ['A','C','E']

    true_answer:
        'C'
    """

    if true_answer in predictions:

        rank = predictions.index(true_answer) + 1

        return 1.0 / rank

    return 0.0


def average_map(scores):
    """
    Compute average MAP@3.
    """

    return float(np.mean(scores))


# =====================================================
# Helpers
# =====================================================

def option_dictionary(row):
    """
    Convert a dataframe row into
    an option dictionary.
    """

    return {
        "A": str(row["A"]),
        "B": str(row["B"]),
        "C": str(row["C"]),
        "D": str(row["D"]),
        "E": str(row["E"]),
    }


def reverse_option_dictionary(option_dict):
    """
    Reverse mapping:
    Text -> Option Letter
    """

    return {
        v: k
        for k, v in option_dict.items()
    }


def ensure_directory(path):
    """
    Create directory if missing.
    """

    os.makedirs(path, exist_ok=True)
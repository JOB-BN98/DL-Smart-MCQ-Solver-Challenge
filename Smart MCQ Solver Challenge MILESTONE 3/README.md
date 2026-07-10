# Smart-MCQ-Solver-Challenge

## Student Information

**Name:** JOB BENNY NAMBERIL
**Student ID:** 22f3000174

---

# Project Title

**Smart MCQ Solver Challenge: Intelligent Multiple Choice Question Answering using NLP, Transformers, RAG, and Fine-Tuned Language Models**

---

# Project Overview

The Smart MCQ Solver Challenge aims to build an intelligent AI system capable of solving complex multiple-choice questions.

Each question consists of:

* A question prompt
* Five answer choices (A, B, C, D, E)

The objective is to predict the **top three most probable answers** in ranked order.

The project progressively explores classical Natural Language Processing techniques, Transformer-based language models, Retrieval-Augmented Generation (RAG), parameter-efficient fine-tuning, and ensemble methods to improve answer ranking performance.

Model performance is evaluated using **Mean Average Precision at 3 (MAP@3)**.

---

# Objectives

The project aims to:

* Preprocess and clean textual data
* Generate text embeddings
* Measure semantic similarity
* Build baseline NLP models
* Explore Transformer architectures
* Implement Retrieval-Augmented Generation (RAG)
* Fine-tune language models using LoRA
* Build ensemble models
* Maximize MAP@3 on the Smart MCQ Solver Challenge dataset

---

# Project Milestones

## Milestone 0 – Environment Setup

* Orientation
* Kaggle setup
* GitHub setup
* Weights & Biases setup
* Development environment configuration

---

## Milestone 1 – NLP Foundation

* Text cleaning
* Tokenization
* Missing value handling
* TF-IDF embeddings
* Word2Vec embeddings
* Cosine similarity
* MAP@3 evaluation

---

## Milestone 2 – Transformers

* Hugging Face Transformers
* Hugging Face Datasets
* BERT architecture
* RoBERTa architecture
* Attention mechanism
* Context-aware embeddings
* Zero-shot classification

---

## Milestone 3 – Retrieval-Augmented Generation

* Introduction to RAG
* Vector database
* Semantic retrieval
* Context augmentation
* Improved reasoning

---

## Milestone 4 – Fine-Tuning

* MCQ data formatting
* LoRA fine-tuning
* Efficient training
* GPU optimization
* Training arguments
* Batch optimization

---

## Milestone 5 – Ensemble Learning

* Top-3 answer ranking
* Logit extraction
* Model ensembling
* Performance optimization
* Final Kaggle submission

---

# Models to be Implemented

The project will include at least three different models:

### Model 1 (Built from Scratch)

* TF-IDF
* Word2Vec
* Cosine Similarity Ranking

### Model 2 (Pretrained)

* BERT
* RoBERTa
* DistilBERT (depending on experimentation)

### Model 3 (Additional Model)

Possible choices include:

* Sentence Transformers
* MiniLM
* RAG Pipeline
* DeBERTa
* MPNet

---

# Evaluation Metrics

Primary Metric

* MAP@3 (Mean Average Precision at 3)

Additional Metrics

* Accuracy
* Macro F1 Score
* Error Analysis

---

# Experiment Tracking

Experiments will be tracked using **Weights & Biases (W&B)**.

Each experiment will log:

* Training loss
* Validation loss
* Accuracy
* MAP@3
* F1 Score
* Hyperparameters

A minimum of **three experimental runs** will be compared.

---

# Repository Structure

```text
project-name/
│
├── notebooks/
│   └── milestone-1.ipynb
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── data/
│
├── models/
│
├── reports/
│   └── milestone-1-report.pdf
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Gensim
* Hugging Face Transformers
* PyTorch
* Weights & Biases
* Jupyter Notebook


# Expected Outcome

The final system will combine classical NLP techniques, Transformer-based language models, Retrieval-Augmented Generation, and ensemble learning to accurately rank the top three answers for each multiple-choice question while maximizing the MAP@3 evaluation score.

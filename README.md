# Knowledge Enhancement in Language Models: Retrieval Augmentation or Adapter-Based Integration?

This repository contains the final project for **INFO7610 – Special Topics in Natural Language Engineering Methods/Tools** at Northeastern University. The final report is provided as **INFO7610_Final_Report.pdf**.

The project presents a controlled comparative study of two representative knowledge enhancement paradigms for language models:

- **Retrieval-Augmented Generation (RAG)** — retrieval-based knowledge augmentation at inference time  
- **Embedding-based knowledge integration (K-Adapter)** — injecting structured knowledge into model parameters via adapter modules  

The primary objective is to evaluate their effectiveness in improving **domain-specific factual recall** and mitigating hallucination under a unified experimental setting.

---

## Project Overview

Large language models (LLMs) exhibit strong linguistic capabilities but often struggle with reliable factual recall, particularly in domain-specific scenarios. This limitation frequently leads to hallucination, where models generate fluent but incorrect statements.

In this project, we compare retrieval-based and parametric knowledge enhancement methods using:

- the **same backbone model**
- the **same domain-specific knowledge graph**
- the **same factual recall evaluation task**

This controlled setup allows for a direct comparison between RAG and K-Adapter, isolating the impact of how external knowledge is incorporated into the model.

---

## Methods

### Baselines

- **BERT-base**
- **BERT-DAPT**: BERT-base further trained with domain-adaptive pretraining (DAPT) using a masked language modeling (MLM) objective on a domain-specific corpus

### Retrieval-Augmented Generation (RAG)

- A domain-specific knowledge graph (KG) is constructed from Wikipedia articles
- Relevant factual triples are retrieved using a **BM25-based retriever**
- Retrieved facts are appended to the input as additional context at inference time
- The augmented input is passed to the BERT-DAPT model for prediction

### Adapter-Based Knowledge Integration (K-Adapter)

- Uses **BERT-DAPT** as a shared backbone
- Introduces two adapter modules:
  - a **knowledge adapter**, trained with supervision from KG triples
  - a **text adapter**, trained with masked language modeling
- Training proceeds in two stages:
  1. Freeze the backbone and train adapters using a joint MLM + KG objective
  2. Unfreeze backbone and adapters and continue MLM training on the domain corpus

---

## Dataset and Knowledge Graph

- **Source**: 1,243 Wikipedia articles from psychology-related categories  
  (e.g., cognitive biases, heuristics, decision-making)
- **Knowledge Graph Construction**:
  - Triples extracted using **REBEL**
  - Followed by post-processing and filtering to remove noisy or low-quality triples
- **Evaluation Data**:
  - Cloze-style factual recall questions derived from KG triples
  - Tail entities are masked and predicted by the model

> **Note:**  
> Raw and processed datasets are **not included** in this repository.  
> All datasets can be regenerated using the provided data processing scripts.

---

## Evaluation

We evaluate model performance using a ranking-based factual recall setup with the following metrics:

- **Accuracy**
- **Mean Reciprocal Rank (MRR)**
- **Average Cosine Similarity**

### Key Findings

- **RAG achieves the strongest and most consistent performance** across all metrics
- **K-Adapter improves over text-only baselines**, but is more sensitive to data scale and knowledge quality
- **DAPT alone provides limited gains**, suggesting misalignment between MLM objectives and factual recall evaluation

Overall, retrieval-based augmentation offers a more robust solution under limited domain-specific data settings, while adapter-based integration remains a promising but more data-dependent alternative.

---

## Repository Structure

```text
bert_dapt/        # Domain-adaptive pretraining (DAPT)
kadapter/         # K-Adapter implementation
datasets_lg/      # Data construction scripts (no data included)
evaluate/         # Evaluation scripts

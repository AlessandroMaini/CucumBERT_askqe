# Fact Coverage Branch

This branch implements a modification to the ASKQE pipeline that enforces a **1:1 correspondence between atomic facts and generated questions**.

## Rationale

The standard ASKQE approach extracts atomic facts from text and generates questions from them, but doesn't guarantee that each atomic fact maps to exactly one question. This branch introduces a **Fact Coverage** constraint that ensures every extracted atomic fact corresponds to precisely one generated question, providing more controlled and predictable question generation behavior.

## Running the Pipeline

### Pipeline with Fact Coverage

To run the ASKQE pipeline with the Fact Coverage extension, use the provided Jupyter notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlessandroMaini/CucumBERT_askqe/blob/fact-coverage/notebooks/askqe_fact_coverage_extension.ipynb)

The notebook includes:
- Setup and configuration for the Fact Coverage pipeline
- Step-by-step execution of question generation with 1:1 fact mapping
- Examples and usage patterns

### Evaluation Analysis

The same notebook (`askqe_fact_coverage_extension.ipynb`) also contains evaluation analysis comparing the Fact Coverage approach against the atomic baseline. This allows you to assess the impact of enforcing the 1:1 correspondence on question quality and coverage metrics.

## Quick Start

1. Open the notebook in Google Colab using the badge above
2. Follow the setup instructions in the notebook
3. Run the pipeline cells to generate questions with Fact Coverage
4. Run the evaluation cells to compare against the atomic baseline

---

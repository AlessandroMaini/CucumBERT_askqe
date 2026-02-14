# CucumBERT_askqe: Extensions to AskQE for Machine Translation Quality Estimation

This repository contains extensions and modifications to the **AskQE: Question Answering as Automatic Evaluation for Machine Translation** framework, originally proposed by Ki et al. (ACL 2025 Findings).

AskQE is a question generation and answering framework designed to detect critical MT errors and provide actionable feedback, helping users decide whether to accept or reject MT outputs even without knowledge of the target language.

## Original Paper

**AskQE: Question Answering as Automatic Evaluation for Machine Translation**  
*Authors: Dayeon Ki, Kevin Duh, Marine Carpuat*

- 📄 [Paper](https://arxiv.org/pdf/2504.11582)
- 💻 [Original Code](https://github.com/dayeonki/askqe)
- 📖 [ACL Anthology](https://aclanthology.org/2025.findings-acl.899/)

---

## Project Extensions

This repository explores three novel extensions to the original AskQE framework, each implemented in a dedicated branch:

### 🔹 [Domain Adaptation](https://github.com/AlessandroMaini/CucumBERT_askqe/tree/domain-adaptation)
Applies the baseline AskQE pipeline to a **Legal domain dataset** instead of the original TICO-19 COVID-19 domain dataset. This extension investigates how well the question generation and answering approach generalizes to specialized domains with distinct terminology and linguistic patterns.

**Key Features:**
- Implementation of the same baseline pipeline
- Adaptation to legal domain translations
- Comparative analysis with TICO-19 results

### 🔹 [Answerability Check](https://github.com/AlessandroMaini/CucumBERT_askqe/tree/answerability-check)
Enhances the Vanilla pipeline by adding a **filtering mechanism** that automatically identifies and discards "unanswerable" questions. This modification improves the quality of generated questions.

**Key Features:**
- Answerability classification module
- Filtering of low-quality questions

### 🔹 [Fact Coverage](https://github.com/AlessandroMaini/CucumBERT_askqe/tree/fact-coverage)
Modifies the NLI (atomic) pipeline by enforcing that **each atomic fact is covered by exactly one question**. This is achieved through careful prompt engineering or multiple model calls, ensuring comprehensive and non-redundant coverage of source sentence information.

**Key Features:**
- One-to-one mapping between atomic facts and questions
- Enhanced fact coverage guarantees
- Reduced redundancy in question generation

---

## Running the Baseline Experiments

This repository provides three Jupyter notebooks for running the baseline AskQE experiments. All notebooks can be executed directly in Google Colab.

### 📓 Notebook 1: Dataset Generation
**Purpose:** Generate synthetically perturbed dataset (ContraTICO) and backtranslations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlessandroMaini/CucumBERT_askqe/blob/main/notebooks/baseline_dataset.ipynb)

This notebook implements:
- Synthetic error injection (synonym, expansion, omission, alteration, etc.)
- Backtranslation using Google Translate API

### 📓 Notebook 2: QG/QA Pipeline Execution
**Purpose:** Execute the Question Generation and Question Answering pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlessandroMaini/CucumBERT_askqe/blob/main/notebooks/baseline_pipeline.ipynb)

This notebook implements:
- Atomic fact extraction using Llama3-70B
- Entailment classification with NLI models
- Question generation conditioned on entailed facts
- Question answering on source and backtranslated MT
- Answer comparison and similarity scoring

### 📓 Notebook 3: Results Analysis
**Purpose:** Perform additional analysis on experimental results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlessandroMaini/CucumBERT_askqe/blob/main/notebooks/askqe_dataset_analysis.ipynb)

This notebook provides:
- Correlation analysis with standard MT evaluation metrics (COMET, BERTScore)
- Performance comparison across perturbation types
- Visualization of results

---

## Repository Structure

```
CucumBERT_askqe/
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── baseline_dataset.ipynb   # Dataset generation
│   ├── baseline_pipeline.ipynb  # QG/QA pipeline
│   └── askqe_dataset_analysis.ipynb  # Results analysis
├── QG/                          # Question Generation code
├── QA/                          # Question Answering code
├── contratico/                  # ContraTICO dataset generation
├── backtranslation/             # Backtranslation utilities
├── evaluation/                  # Evaluation metrics and baselines
├── data/                        # Datasets and processed data
└── requirements.txt             # Python dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AlessandroMaini/CucumBERT_askqe.git
cd CucumBERT_askqe

# Install dependencies
pip install -r requirements.txt
```

**Note:** You will need to set up API keys for:
- Groq API (for Llama3-70B)

---
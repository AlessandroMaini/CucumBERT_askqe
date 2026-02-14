# AskQE: Legal Domain Evaluation

**This branch focuses on evaluating AskQE performance on the legal domain.** This repository extends the original AskQE framework (from the ACL 2025 Findings paper) to assess its effectiveness on legal texts, which present unique challenges including high lexical overlap, rigid terminology, and complex conditional structures.

## Overview
This branch evaluates the AskQE metric's performance and stability when transitioned from the medical/COVID-19 domain to a highly technical legal domain. While the original framework was tested on informative/narrative corpora, legal texts present unique challenges that test the robustness of question-answering based MT evaluation.

## Legal Domain Dataset

This branch evaluates AskQE on the **OPUS-DGT** corpus containing legal documents from the European Commission's Directorate-General for Translation. Legal texts present unique challenges:
- High lexical overlap between source and target
- Rigid, technical terminology
- Complex conditional logic (e.g., "unless otherwise provided")
- Less paraphrasing variability than medical/general domains

**Source Reference:** Jorg Tiedemann. "Parallel Data, Tools and Interfaces in OPUS." In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12), Istanbul, Turkey, 2012.

### Dataset Generation

Scripts to generate the legal domain dataset are located in [**data folder**](data/code/):
- [`extract_ids.py`](data/code/extract_ids.py): Extract document IDs from the OPUS-DGT corpus
- `get_dataset_{language_pair}.py`: Download and process the new dataset from internet
- [`sub_dataset.py`](data/code/sub_dataset.py): Create subset of the dataset for evaluation

For detailed information on dataset generation, see the [README](data/code/DATASET-PREPROCESSING-README).

### Processed Data
- Data for each language pair: `data/processed/{language pair}.json`

## Evaluation Notebook

The comprehensive evaluation and analysis can be found in the `notebooks` folder:
- [**askqe_domain_adaptation_eval.ipynb**](notebooks/askqe_domain_adaptation_eval.ipynb): Complete evaluation notebook with:
  - Legal domain dataset analytics
  - Question generation analysis (vanilla vs atomic pipelines)
  - String comparison metrics (F1, BLEU, chrF)
  - Pearson correlation analysis
  - Silhouette score analysis
  - Results visualization and key findings

This notebook provides detailed insights into AskQE's performance on legal domain translations.

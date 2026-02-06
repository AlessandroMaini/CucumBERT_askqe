# Dataset filtering (Question Generation)

When generating JSON data from parallel corpora, we keep only English sentences that meet all filters below.

## Dataset Generation Scripts

Two dataset generation scripts are available for different language pairs:

- **get_dataset_en-es.py** - Generates bilingual English-Spanish dataset (`en-es.jsonl`)
- **get_dataset_en-fr.py** - Generates bilingual English-French dataset (`en-fr.jsonl`)

Both scripts download data from the OPUS-DGT corpus, extract parallel sentences, apply the filters below, and output JSONL format with `id`, `en`, and the target language key (`es` or `fr`).

## Dataset Source Reference

OPUS dataset reference:

Jorg Tiedemann. "Parallel Data, Tools and Interfaces in OPUS." In *Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)*, Istanbul, Turkey, May 23-25, 2012. European Language Resources Association (ELRA). ISBN: 978-2-9517408-7-7.

## English sentence filters
- Length between 50 and 250 characters.
- At least 8 words.
- Alphabetic ratio ≥ 0.7.
- Not a question (no trailing `?` and no leading WH‑word).
- No URLs or emails.
- No long numeric runs (6+ digits).
- Not an all‑caps/heading‑like line (uppercase ratio ≤ 0.3).


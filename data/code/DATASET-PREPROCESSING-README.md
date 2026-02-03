# Dataset filtering (Question Generation)

When generating JSON data from parallel corpora using get_dataset.py, we keep only English sentences that meet all filters below.

## English sentence filters
- Length between 50 and 250 characters.
- At least 8 words.
- Alphabetic ratio ≥ 0.7.
- Not a question (no trailing `?` and no leading WH‑word).
- No URLs or emails.
- No long numeric runs (6+ digits).
- Not an all‑caps/heading‑like line (uppercase ratio ≤ 0.3).


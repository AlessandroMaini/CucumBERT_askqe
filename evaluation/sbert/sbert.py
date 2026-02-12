import json
import nltk
import argparse
import csv
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


nltk.download("punkt")

parser = argparse.ArgumentParser(description="Evaluate QA results using SBERT similarity")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'qwen3-4b')")
parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
args = parser.parse_args()

# Get the workspace root (2 levels up from this script in evaluation/sbert/)
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent.parent

# Language configurations: (language_code, is_mini)
language_configs = [
    ("es", False),
    ("es", True),   # mini version
    ("fr", False),
    ("fr", True),   # mini version
]

pipelines = ["vanilla", "atomic", "anscheck"]
perturbations = ["synonym", "expansion_noimpact",
                 "omission", "alteration"]
check_variants = ["longformer", "electra"]


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# Resolve output file path
output_path = Path(args.output_file)
if not output_path.is_absolute():
    output_path = workspace_root / args.output_file

# Ensure output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Write CSV header
with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["language", "is_mini", "perturbation", "pipeline", "check_variant", "cosine_similarity", "num_comparison"])

# Process each configuration
for language, is_mini in language_configs:
    for pipeline in pipelines:
        # Determine which check_variants to iterate (only for anscheck)
        variants_to_process = check_variants if pipeline == "anscheck" else [None]
        
        for check_variant in variants_to_process:
            for perturbation in perturbations:
                lang_label = f"{language}{'-mini' if is_mini else ''}"
                print(f"Language: {lang_label}")
                print(f"Pipeline: {pipeline}")
                if check_variant:
                    print(f"Check Variant: {check_variant}")
                print(f"Perturbation: {perturbation}")

                # Build dataset name
                dataset_name = f"en-{language}{'-mini' if is_mini else ''}"
                
                # Build file paths
                if pipeline == "anscheck" and check_variant:
                    if is_mini:
                        predicted_filename = f"{language}-anscheck-{check_variant}-{perturbation}-mini.jsonl"
                        reference_filename = f"en-anscheck-{check_variant}-mini.jsonl"
                    else:
                        predicted_filename = f"{language}-anscheck-{check_variant}-{perturbation}.jsonl"
                        reference_filename = f"en-anscheck-{check_variant}.jsonl"
                else:
                    if is_mini:
                        predicted_filename = f"{language}-{pipeline}-{perturbation}-mini.jsonl"
                        reference_filename = f"en-{pipeline}-mini.jsonl"
                    else:
                        predicted_filename = f"{language}-{pipeline}-{perturbation}.jsonl"
                        reference_filename = f"en-{pipeline}.jsonl"
                
                predicted_file = workspace_root / "QA" / args.model / predicted_filename
                reference_file = workspace_root / "QA" / args.model / reference_filename

                total_cosine_similarity = 0
                num_comparisons = 0
                results_list = []

                try:
                    # Load reference file into a dict keyed by ID
                    ref_by_id = {}
                    with open(reference_file, "r", encoding="utf-8") as ref_file:
                        for ref_line in ref_file:
                            try:
                                ref_data = json.loads(ref_line)
                                ref_id = ref_data.get("id")
                                if ref_id is not None:
                                    ref_by_id[ref_id] = ref_data
                            except json.JSONDecodeError:
                                continue

                    with open(predicted_file, "r", encoding="utf-8") as pred_file:
                        for pred_line in pred_file:
                            try:
                                pred_data = json.loads(pred_line)
                                pred_id = pred_data.get("id")
                                if pred_id is None or pred_id not in ref_by_id:
                                    continue
                                ref_data = ref_by_id[pred_id]

                                predicted_answers = pred_data.get("answers", [])
                                reference_answers = ref_data.get("answers", [])

                                if isinstance(predicted_answers, str):
                                    try:
                                        predicted_answers = json.loads(predicted_answers)
                                    except json.JSONDecodeError:
                                        continue

                                if isinstance(reference_answers, str):
                                    try:
                                        reference_answers = json.loads(reference_answers)
                                    except json.JSONDecodeError:
                                        continue

                                if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                                    continue
                                if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                                    continue
                                
                                # Store cosine similarities for this record
                                cosine_similarities = []
                                
                                for pred, ref in zip(predicted_answers, reference_answers):
                                    if not isinstance(pred, str) or not isinstance(ref, str):
                                        continue
                                    if pred.strip() == "" or ref.strip() == "":
                                        continue

                                    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
                                    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

                                    with torch.no_grad():
                                        pred_output = model(**encoded_pred)
                                        ref_output = model(**encoded_ref)

                                    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                                    pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                                    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                                    ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                                    cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                                    cosine_similarities.append(cos_sim)
                                    total_cosine_similarity += cos_sim
                                    num_comparisons += 1
                                
                                # Collect record with scores
                                if cosine_similarities:
                                    output_record = pred_data.copy()
                                    output_record["sbert_scores"] = cosine_similarities
                                    output_record["sbert_score"] = sum(cosine_similarities) / len(cosine_similarities)
                                    results_list.append(output_record)

                            except json.JSONDecodeError as e:
                                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                                continue

                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    continue
                
                # Structure: sbert/{dataset}/{pipeline}/[anscheck-type if applicable-]{perturbation}.jsonl
                if num_comparisons > 0 and results_list:
                    avg_cosine_similarity = total_cosine_similarity / num_comparisons

                    # Build output JSONL path and write only if we have results
                    if pipeline == "anscheck" and check_variant:
                        output_jsonl_dir = script_dir / dataset_name / pipeline
                        output_jsonl_filename = f"{check_variant}-{perturbation}.jsonl"
                    else:
                        output_jsonl_dir = script_dir / dataset_name / pipeline
                        output_jsonl_filename = f"{perturbation}.jsonl"
                    
                    output_jsonl_dir.mkdir(parents=True, exist_ok=True)
                    output_jsonl_path = output_jsonl_dir / output_jsonl_filename

                    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_out:
                        for record in results_list:
                            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    print("-" * 80)
                    print("Average Scores:")
                    print(f"Num comparisons: {num_comparisons}")
                    print(f"Cosine Similarity: {avg_cosine_similarity:.3f}")
                    print(f"Detailed results saved to: {output_jsonl_path}")
                    print("=" * 80)

                    # Append results to CSV
                    with open(output_path, mode="a", newline="", encoding="utf-8") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([language, is_mini, perturbation, pipeline, check_variant if check_variant else "N/A", avg_cosine_similarity, num_comparisons])

                else:
                    print("No valid comparisons found in the JSONL files.")
                    print("-" * 80)
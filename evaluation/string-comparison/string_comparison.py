import json
import nltk
import argparse
from pathlib import Path
from utils import compare_answers

nltk.download("punkt")

parser = argparse.ArgumentParser(description="Evaluate QA results using string comparison metrics")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'llama-70b', 'qwen3-4b')")
args = parser.parse_args()

# Get the workspace root (2 levels up from this script in evaluation/string-comparison/)
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
perturbations = ["alteration", "expansion_noimpact", "omission", "synonym"]
check_variants = ["longformer", "electra"]


for language, is_mini in language_configs:
    for pipeline in pipelines:
        # Determine which check_variants to iterate (only for anscheck)
        variants_to_process = check_variants if pipeline == "anscheck" else [None]
        
        for check_variant in variants_to_process:
            for perturbation in perturbations:
                lang_label = f"{language}{'-mini' if is_mini else ''}"
                print(f"Processing: {lang_label} | {pipeline}", end="")
                if check_variant:
                    print(f" | {check_variant}", end="")
                print(f" | {perturbation}")
                
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

                                row_scores = []
                                for pred, ref in zip(predicted_answers, reference_answers):
                                    # Ensure both pred and ref are strings
                                    if not isinstance(pred, str) or not isinstance(ref, str):
                                        continue
                                    if pred.strip() == "" or ref.strip() == "":
                                        continue
                                        
                                    f1, EM, chrf, bleu = compare_answers(pred, ref)
                                    row_scores.append({
                                        "f1": f1,
                                        "em": EM,
                                        "chrf": chrf,
                                        "bleu": bleu
                                    })

                                # Only save if we have valid scores
                                if not row_scores:
                                    continue

                                # Calculate averaged scores
                                avg_f1 = sum(score["f1"] for score in row_scores) / len(row_scores)
                                avg_em = sum(score["em"] for score in row_scores) / len(row_scores)
                                avg_chrf = sum(score["chrf"] for score in row_scores) / len(row_scores)
                                avg_bleu = sum(score["bleu"] for score in row_scores) / len(row_scores)

                                # Save per-row result with all original data
                                row_data = pred_data.copy()
                                row_data["scores"] = row_scores
                                row_data["avg_f1"] = avg_f1
                                row_data["avg_em"] = avg_em
                                row_data["avg_chrf"] = avg_chrf
                                row_data["avg_bleu"] = avg_bleu
                                results_list.append(row_data)

                            except json.JSONDecodeError as e:
                                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                                continue

                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    continue

                # Only create output file if we have valid results
                if results_list:
                    # Build output path
                    if is_mini:
                        base_dir = script_dir / f"en-{language}-mini"
                    else:
                        base_dir = script_dir / f"en-{language}"
                    
                    if pipeline == "anscheck" and check_variant:
                        output_dir = base_dir / "anscheck"
                        output_path = output_dir / f"{check_variant}-{perturbation}.jsonl"
                    else:
                        output_dir = base_dir / pipeline
                        output_path = output_dir / f"{perturbation}.jsonl"
                    
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, "w", encoding="utf-8") as jsonl_file:
                        for row in results_list:
                            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
                    print(f"Saved results to: {output_path}")
                else:
                    print("No valid results found.")
                
                print("-" * 80)
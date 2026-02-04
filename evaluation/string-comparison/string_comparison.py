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
    ("hi", False),
    ("tl", False),
    ("zh", False)
]

pipelines = ["vanilla", "semantic", "atomic", "anscheck"]
perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]
check_variants = ["longformer", "electra", "electra-null"]


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
                    with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
                        for pred_line, ref_line in zip(pred_file, ref_file):
                            try:
                                pred_data = json.loads(pred_line)
                                ref_data = json.loads(ref_line)

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

                                # Save per-row result
                                row_data = {
                                    "id": pred_data.get("id", "unknown"),
                                    "en": pred_data.get("en", "unknown"),
                                    "scores": row_scores
                                }
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
                    if pipeline == "anscheck" and check_variant:
                        if is_mini:
                            output_dir = script_dir / f"en-{language}-mini" / "anscheck"
                        else:
                            output_dir = script_dir / f"en-{language}" / "anscheck"
                        output_path = output_dir / f"{check_variant}-{perturbation}.jsonl"
                    else:
                        if is_mini:
                            output_dir = script_dir / f"en-{language}-mini"
                        else:
                            output_dir = script_dir / f"en-{language}"
                        output_path = output_dir / f"{perturbation}.jsonl"
                    
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, "w", encoding="utf-8") as jsonl_file:
                        for row in results_list:
                            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
                    print(f"Saved results to: {output_path}")
                else:
                    print("No valid results found.")
                
                print("-" * 80)

import json
import torch
import argparse
from pathlib import Path
from bert_score import score


def main():
    parser = argparse.ArgumentParser(description="Evaluate backtranslation quality using BERTScore")
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-xlarge-mnli",
                        help="BERTScore model to use (default: microsoft/deberta-xlarge-mnli)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for scoring (default: 2)")
    args = parser.parse_args()
    
    # Get the workspace root (2 levels up from this script in evaluation/bt-score/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                     "intensifier", "expansion_impact", "omission", "alteration"]

    for language, is_mini in language_configs:
        for perturbation in perturbations:
            lang_label = f"{language}{'-mini' if is_mini else ''}"
            print(f"Processing - Language: {lang_label}, Perturbation: {perturbation}")

            # Build file path
            if is_mini:
                input_file = workspace_root / "backtranslation" / f"en-{language}-mini" / f"bt-{perturbation}.jsonl"
                output_dir = script_dir / f"en-{language}-mini"
            else:
                input_file = workspace_root / "backtranslation" / f"en-{language}" / f"bt-{perturbation}.jsonl"
                output_dir = script_dir / f"en-{language}"
            
            output_file = output_dir / f"bt-{perturbation}_bertscore.jsonl"
            
            # Ensure input file exists
            if not input_file.exists():
                print(f"Input file not found: {input_file}")
                continue
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Collect all data for batch processing
            src_sentences = []
            mt_sentences = []
            raw_data = []
            
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if "en" in item and f"bt_pert_{language}" in item:
                            src_sentences.append(item["en"])
                            mt_sentences.append(item[f"bt_pert_{language}"])
                            raw_data.append(item)
                    except json.JSONDecodeError:
                        continue

            if not src_sentences:
                print(f"No valid data in {input_file}")
                continue

            # Run BERTScore
            print(f"Running BERTScore on {len(src_sentences)} samples...")
            P, R, F1 = score(mt_sentences, src_sentences, lang="en",
                           model_type=args.model_type,
                           device=device,
                           batch_size=args.batch_size)

            average_score = F1.mean().item() if len(F1) > 0 else 0

            # Map scores back to records
            with open(output_file, "w", encoding="utf-8") as out_f:
                for item, f1_score in zip(raw_data, F1):
                    item["bertscore_f1"] = f1_score.item()
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Average BERTScore (F1): {average_score:.4f}")
            print(f"Saved results to {output_file}")
            print("="*80)


if __name__ == "__main__":
    main()
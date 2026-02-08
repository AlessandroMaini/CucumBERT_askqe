from comet import download_model, load_from_checkpoint
import json
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate MT quality using XCOMET")
    parser.add_argument("--model", type=str, default="Unbabel/wmt22-cometkiwi-da", 
                        help="COMET model to use (default: Unbabel/wmt22-cometkiwi-da)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for prediction (default: 16)")
    args = parser.parse_args()
    
    # Get the workspace root (2 levels up from this script in evaluation/xcomet/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Use specified model
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    # Check if GPU is available
    gpus = 1 if torch.cuda.is_available() else 0

    # Language configurations: (language_code, is_mini)
    language_configs = [
        ("es", False),
        ("es", True),   # mini version
        ("fr", False),
        ("fr", True),   # mini version
    ]

    perturbations = ["synonym", "expansion_noimpact",
                     "omission", "alteration"]

    for language, is_mini in language_configs:
        for perturbation in perturbations:
            lang_label = f"{language}{'-mini' if is_mini else ''}"
            print(f"Processing - Language: {lang_label}, Perturbation: {perturbation}")

            # Build file path - only need the contratico file
            if is_mini:
                input_file = workspace_root / "contratico" / f"en-{language}-mini" / f"{perturbation}.jsonl"
                output_dir = script_dir / f"en-{language}-mini"
            else:
                input_file = workspace_root / "contratico" / f"en-{language}" / f"{perturbation}.jsonl"
                output_dir = script_dir / f"en-{language}"
            
            output_file = output_dir / f"{perturbation}.jsonl"
            
            # Ensure input file exists
            if not input_file.exists():
                print(f"Input file not found: {input_file}")
                continue
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Collect all data for batch processing
            records_to_process = []
            comet_input = []

            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        src = data.get('en', '')  # Source is the original English
                        mt = data.get(f'pert_{language}', '')  # Prediction is the perturbed text

                        if src and mt:
                            records_to_process.append(data)
                            comet_input.append({
                                "src": src,
                                "mt": mt
                            })
                    except json.JSONDecodeError:
                        continue

            # Run Prediction in BATCHES (Much faster)
            if comet_input:
                print(f"Running COMET on {len(comet_input)} samples...")
                # Use batch_size from arguments
                model_output = model.predict(comet_input, batch_size=args.batch_size, gpus=gpus)

                scores = model_output.scores
                # Use getattr for metadata as some models don't return error_spans
                metadata = getattr(model_output, "metadata", {})
                error_spans = metadata.get("error_spans", [[]] * len(scores))

                # Map scores back to records
                for i, record in enumerate(records_to_process):
                    record["xcomet_annotation"] = {
                        "segment_score": round(scores[i], 3),
                        "error_spans": error_spans[i]
                    }

            # Write results
            with open(output_file, "w", encoding="utf-8") as out_f:
                for record in records_to_process:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

            print(f"Saved results to {output_file}")
            print("="*80)

if __name__ == "__main__":
    main()
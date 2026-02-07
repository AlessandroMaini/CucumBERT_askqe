import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Define perturbation types
PERTURBATIONS = ["synonym", "alteration", "spelling", "expansion_impact", "expansion_noimpact"]

# Define datasets
DATASETS = ["en-es", "en-es-mini", "en-fr", "en-fr-mini", "en-hi", "en-tl", "en-zh"]

# Define pipelines
PIPELINES = ["vanilla", "semantic", "atomic", "anscheck"]

# Define anscheck types
ANSCHECK_TYPES = ["longformer", "electra", "electra-null"]

# Define metric configurations
# Standard metrics: stored as {metric}/{dataset}/{perturbation}.jsonl
STANDARD_METRICS = {
    "xcomet": {
        "base_path": "evaluation/xcomet",
        "field_extractor": lambda data: data.get("xcomet_annotation", {}).get("segment_score")
    },
    "bt_score": {
        "base_path": "evaluation/bt-score",
        "field_extractor": lambda data: data.get("bertscore_f1")
    }
}

# AskQE metrics: stored as {metric}/{dataset}/{pipeline}/[{anscheck_type}-]{perturbation}.jsonl
ASKQE_METRICS = {
    "f1": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda data: data.get("avg_f1") if data.get("avg_f1") is not None else (np.mean([score["f1"] for score in data.get("scores", [])]) if data.get("scores") else None)
    },
    "em": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda data: data.get("avg_em") if data.get("avg_em") is not None else (np.mean([1.0 if score["em"] else 0.0 for score in data.get("scores", [])]) if data.get("scores") else None)
    },
    "bleu": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda data: data.get("avg_bleu") if data.get("avg_bleu") is not None else (np.mean([score["bleu"] for score in data.get("scores", [])]) if data.get("scores") else None)
    },
    "chrf": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda data: data.get("avg_chrf") if data.get("avg_chrf") is not None else (np.mean([score["chrf"] for score in data.get("scores", [])]) if data.get("scores") else None)
    },
    "sbert": {
        "base_path": "evaluation/sbert",
        "field_extractor": lambda data: data.get("sbert_score")
    }
}


def load_standard_metric_data(metric_name, dataset, workspace_root):
    """Load Standard metric data (xcomet, bt_score) from JSONL files."""
    config = STANDARD_METRICS.get(metric_name)
    if not config:
        raise ValueError(f"Unknown standard metric: {metric_name}")
    
    # Standard metrics path: {metric}/{dataset}/{perturbation}.jsonl
    base_path = workspace_root / config["base_path"] / dataset
    
    all_data = {}
    files_found = 0
    
    for pert in PERTURBATIONS:
        # Special naming for bt_score files
        if metric_name == "bt_score":
            file_path = base_path / f"bt-{pert}_bertscore.jsonl"
        else:
            file_path = base_path / f"{pert}.jsonl"
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        files_found += 1
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        record_id = data.get("id")
                        if record_id:
                            score = config["field_extractor"](data)
                            if score is not None:
                                composite_key = f"{record_id}_{pert}"
                                all_data[composite_key] = score
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {e}")
                        continue
    
    if files_found == 0:
        raise FileNotFoundError(f"No files found for metric {metric_name} in {base_path}")
    
    return all_data


def load_askqe_metric_data(metric_name, dataset, pipeline, anscheck_type, workspace_root):
    """Load AskQE metric data (f1, em, chrf, bleu, sbert) from JSONL files."""
    config = ASKQE_METRICS.get(metric_name)
    if not config:
        raise ValueError(f"Unknown AskQE metric: {metric_name}")
    
    # AskQE metrics path: {metric}/{dataset}/{pipeline}/[{anscheck_type}-]{perturbation}.jsonl
    base_path = workspace_root / config["base_path"] / dataset / pipeline
    
    all_data = {}
    files_found = 0
    
    for pert in PERTURBATIONS:
        # For anscheck pipeline, files are prefixed with anscheck type
        if pipeline == "anscheck":
            if not anscheck_type:
                raise ValueError(f"Anscheck pipeline requires anscheck_type")
            file_path = base_path / f"{anscheck_type}-{pert}.jsonl"
        else:
            file_path = base_path / f"{pert}.jsonl"
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        files_found += 1
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        record_id = data.get("id")
                        if record_id:
                            score = config["field_extractor"](data)
                            if score is not None:
                                composite_key = f"{record_id}_{pert}"
                                all_data[composite_key] = score
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {e}")
                        continue
    
    if files_found == 0:
        raise FileNotFoundError(f"No files found for metric {metric_name} in {base_path}")
    
    return all_data


def compute_correlation(metric1_data, metric2_data):
    """Compute Pearson correlation between two metrics."""
    # Find common IDs
    common_ids = sorted(set(metric1_data.keys()) & set(metric2_data.keys()))
    
    if not common_ids:
        raise ValueError("No common sentences found between the two metrics")
    
    # Extract values for common IDs
    values1 = [metric1_data[id] for id in common_ids]
    values2 = [metric2_data[id] for id in common_ids]
    
    # Compute Pearson correlation
    correlation, p_value = pearsonr(values1, values2)
    
    return correlation, p_value, len(common_ids), common_ids


def main():
    parser = argparse.ArgumentParser(
        description="Compute Pearson correlation between an AskQE metric and a Standard metric."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset identifier (e.g., en-es-mini, en-fr, en-es)"
    )
    
    parser.add_argument(
        "--askqe_metric",
        type=str,
        required=True,
        choices=list(ASKQE_METRICS.keys()),
        help="AskQE metric to compare (f1, em, bleu, chrf, sbert)"
    )
    
    parser.add_argument(
        "--standard_metric",
        type=str,
        required=True,
        choices=list(STANDARD_METRICS.keys()),
        help="Standard metric to compare (xcomet, bt_score)"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=PIPELINES,
        help="Pipeline for AskQE metric (vanilla, semantic, atomic, anscheck)"
    )
    
    parser.add_argument(
        "--anscheck_type",
        type=str,
        choices=ANSCHECK_TYPES,
        help="Answerability check type (required if pipeline is anscheck)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, defaults to evaluation/pearson-correlation/results/)"
    )
    
    args = parser.parse_args()
    
    # Validate anscheck requirements
    if args.pipeline == "anscheck" and not args.anscheck_type:
        parser.error("--anscheck_type is required when pipeline is 'anscheck'")
    
    # Get workspace root
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    print(f"Computing Pearson Correlation")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"AskQE Metric: {args.askqe_metric}")
    print(f"Standard Metric: {args.standard_metric}")
    print(f"Pipeline: {args.pipeline}")
    if args.anscheck_type:
        print(f"Anscheck Type: {args.anscheck_type}")
    print()
    
    try:
        print(f"Loading data for AskQE metric ({args.askqe_metric})...")
        askqe_data = load_askqe_metric_data(
            args.askqe_metric, 
            args.dataset, 
            args.pipeline,
            args.anscheck_type,
            workspace_root
        )
        print(f"  Loaded {len(askqe_data)} records")
        
        print(f"Loading data for Standard metric ({args.standard_metric})...")
        standard_data = load_standard_metric_data(
            args.standard_metric,
            args.dataset,
            workspace_root
        )
        print(f"  Loaded {len(standard_data)} records")
        
        print("\nComputing Pearson correlation...")
        correlation, p_value, n_samples, common_ids = compute_correlation(askqe_data, standard_data)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Pearson correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Number of samples: {n_samples}")
        print(f"{'='*60}")
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = script_dir / "results"
            output_dir.mkdir(exist_ok=True)
            
            # Build descriptive filename
            askqe_name = f"{args.askqe_metric}-{args.pipeline}"
            if args.anscheck_type:
                askqe_name += f"-{args.anscheck_type}"
            
            output_path = output_dir / f"{args.dataset}_{askqe_name}_vs_{args.standard_metric}.txt"
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Pearson Correlation Analysis\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"AskQE Metric: {args.askqe_metric} (pipeline: {args.pipeline}")
            if args.anscheck_type:
                f.write(f", anscheck: {args.anscheck_type}")
            f.write(f")\n")
            f.write(f"Standard Metric: {args.standard_metric}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Pearson correlation: {correlation:.4f}\n")
            f.write(f"  P-value: {p_value:.4e}\n")
            f.write(f"  Number of samples: {n_samples}\n\n")
            f.write(f"Common keys ({len(common_ids)}):\n")
            for idx, key in enumerate(common_ids, 1):
                f.write(f"  {idx}. {key} (AskQE: {askqe_data[key]:.4f}, Standard: {standard_data[key]:.4f})\n")
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

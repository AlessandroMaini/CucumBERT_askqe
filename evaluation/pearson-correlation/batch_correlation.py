"""
Batch script to compute multiple Pearson correlations for a dataset.
This script runs all combinations of AskQE metrics, Standard metrics, and pipelines.
"""

import subprocess
import sys
from pathlib import Path

# Define all possible values
ASKQE_METRICS = ["f1", "em", "bleu", "chrf", "sbert"]
STANDARD_METRICS = ["xcomet", "bt_score"]
PIPELINES = ["vanilla", "semantic", "atomic", "anscheck"]
ANSCHECK_TYPES = ["longformer", "electra", "electra-null"]


def run_correlation(dataset, askqe_metric, standard_metric, pipeline, anscheck_type=None):
    """Run a single correlation analysis."""
    cmd = [
        sys.executable,
        "compute_correlation.py",
        "--dataset", dataset,
        "--askqe_metric", askqe_metric,
        "--standard_metric", standard_metric,
        "--pipeline", pipeline
    ]
    
    if anscheck_type:
        cmd.extend(["--anscheck_type", anscheck_type])
    
    pipeline_desc = f"{pipeline}" + (f"-{anscheck_type}" if anscheck_type else "")
    print(f"\n{'='*60}")
    print(f"Running: {askqe_metric} ({pipeline_desc}) vs {standard_metric}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running correlation: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_correlation.py DATASET")
        print("Example: python batch_correlation.py en-es-mini")
        print("\nThis will compute correlations for all combinations of:")
        print("  - AskQE metrics: f1, em, bleu, chrf, sbert")
        print("  - Standard metrics: xcomet, bt_score")
        print("  - Pipelines: vanilla, semantic, atomic, anscheck (with all anscheck types)")
        sys.exit(1)
    
    dataset = sys.argv[1]
    
    print(f"Computing correlations for dataset: {dataset}")
    print(f"This will run {len(ASKQE_METRICS)} AskQE metrics × {len(STANDARD_METRICS)} standard metrics × {len(PIPELINES) + 2} pipeline configurations")
    print(f"Total: {len(ASKQE_METRICS) * len(STANDARD_METRICS) * (len(PIPELINES) + 2)} correlation analyses\n")
    
    successful = 0
    failed = 0
    total = 0
    
    # Iterate through all combinations
    for askqe_metric in ASKQE_METRICS:
        for standard_metric in STANDARD_METRICS:
            # Regular pipelines (vanilla, semantic, atomic)
            for pipeline in ["vanilla", "semantic", "atomic"]:
                total += 1
                if run_correlation(dataset, askqe_metric, standard_metric, pipeline):
                    successful += 1
                else:
                    failed += 1
            
            # Anscheck pipeline with all anscheck types
            for anscheck_type in ANSCHECK_TYPES:
                total += 1
                if run_correlation(dataset, askqe_metric, standard_metric, "anscheck", anscheck_type):
                    successful += 1
                else:
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

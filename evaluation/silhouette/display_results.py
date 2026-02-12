"""Utility script to display and compare silhouette-score results."""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_results(filepath: Path) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def display_results(results: Dict):
    """Display results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Results for: {results['target_lang']}")
    print('='*80)
    
    print(f"\nCritical Perturbations: {', '.join(results['critical_perturbations'])}")
    print(f"Minimal Perturbations: {', '.join(results['minimal_perturbations'])}")

    def _print_metric(name, scores):
        sil = scores.get('silhouette_score')
        n = scores.get('n_samples', '')
        sil_str = f"{sil:.6f}" if sil is not None else 'N/A'
        print(f"  {name.upper():<12} {sil_str:>12} {n:>8}")

    # --- Pipeline-dependent AskQE metrics ---
    for pipeline_result in results['results']:
        pipeline_name = pipeline_result['pipeline']
        anscheck_type = pipeline_result.get('anscheck_type')

        if anscheck_type:
            pipeline_name = f"{pipeline_name} ({anscheck_type})"

        print(f"\n{'-'*60}")
        print(f"Pipeline: {pipeline_name}  (AskQE metrics)")
        print('-'*60)

        print(f"\n  {'Metric':<12} {'Silhouette':>12} {'N':>8}")
        print(f"  {'-'*34}")

        for m, scores in pipeline_result['metrics'].items():
            _print_metric(m, scores)

    # --- Pipeline-independent MT metrics ---
    mt_results = results.get('mt_results', {})
    if mt_results:
        print(f"\n{'-'*60}")
        print(f"Standard MT metrics (pipeline-independent)")
        print('-'*60)

        print(f"\n  {'Metric':<12} {'Silhouette':>12} {'N':>8}")
        print(f"  {'-'*34}")

        for m, scores in mt_results.items():
            _print_metric(m, scores)


def compare_pipelines(results: Dict):
    """Compare results across different pipelines."""
    print(f"\n{'='*80}")
    print(f"Pipeline Comparison for {results['target_lang']}")
    print('='*80)
    
    # Extract pipeline names
    pipelines = []
    for r in results['results']:
        name = r['pipeline']
        if r.get('anscheck_type'):
            name += f" ({r['anscheck_type']})"
        pipelines.append(name)
    
    # Compare AskQE metrics across pipelines
    if results['results']:
        metrics = list(results['results'][0]['metrics'].keys())

        for metric in metrics:
            print(f"\n{metric.upper()} (AskQE):")
            print('-' * 55)

            for idx, pipeline_result in enumerate(results['results']):
                scores = pipeline_result['metrics'].get(metric, {})
                sil = scores.get('silhouette_score')
                n = scores.get('n_samples', '')
                sil_str = f"{sil:.6f}" if sil is not None else 'N/A'
                print(f"  {pipelines[idx]:<30} {sil_str:>12}  (n={n})")

    # Show MT metrics (same for all pipelines)
    mt_results = results.get('mt_results', {})
    if mt_results:
        print(f"\nStandard MT metrics (pipeline-independent):")
        print('-' * 55)
        for metric, scores in mt_results.items():
            sil = scores.get('silhouette_score')
            n = scores.get('n_samples', '')
            sil_str = f"{sil:.6f}" if sil is not None else 'N/A'
            print(f"  {metric.upper():<30} {sil_str:>12}  (n={n})")


def main():
    parser = argparse.ArgumentParser(
        description='Display and analyze silhouette results'
    )
    parser.add_argument(
        'result_file',
        type=str,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show pipeline comparison'
    )
    
    args = parser.parse_args()
    
    # Get workspace root (2 levels up from this script in evaluation/silhouette/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    # Resolve result file path relative to workspace root if not absolute
    result_path = Path(args.result_file)
    if not result_path.is_absolute():
        result_path = workspace_root / result_path
    
    if not result_path.exists():
        print(f"Error: File not found: {result_path}")
        return
    
    results = load_results(result_path)
    display_results(results)
    
    if args.compare:
        compare_pipelines(results)


if __name__ == '__main__':
    main()
"""
Utility script to display and compare within-between results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


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
    
    for pipeline_result in results['results']:
        pipeline_name = pipeline_result['pipeline']
        anscheck_type = pipeline_result.get('anscheck_type')
        
        if anscheck_type:
            pipeline_name = f"{pipeline_name} ({anscheck_type})"
        
        print(f"\n{'-'*80}")
        print(f"Pipeline: {pipeline_name}")
        print('-'*80)
        
        # Print header
        print(f"\n{'Metric':<10} | {'SILHOUETTE':<22}")
        print('-' * 80)
        
        for metric, scores in pipeline_result['metrics'].items():
            silhouette = scores["silhouette_score"]

            print(f"{metric.upper():<10} | {silhouette:>14.6f}")


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
    
    # For each metric, compare across pipelines
    if results['results']:
        metrics = list(results['results'][0]['metrics'].keys())
        
        for metric in metrics:
            print(f"\n{metric.upper()} - Silhouette:")
            print('-' * 60)
            
            for idx, pipeline_result in enumerate(results['results']):
                scores = pipeline_result['metrics'][metric]
                silhouette = scores['silhouette_score']
                
                print(f"  {pipelines[idx]:<30} | Ratio: {silhouette:>10.4f}")


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
    
    # Get workspace root (2 levels up from this script in evaluation/within-between/)
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

"""
Compute within-category and between-category differences for AskQE metrics.

This script calculates two types of mean squared difference scores:
- DIFF_BETWEEN: Average squared difference between critical and minimal perturbations
- DIFF_WITHIN: Average squared difference within the same category (critical or minimal)

Categories:
- CRITICAL: omission, expansion_impact, alteration
- MINIMAL: spelling, synonym, expansion_noimpact, word_order

The script reads evaluation results from:
- evaluation/string-comparison/{lang}/{pipeline}/: for f1, em, chrf, bleu metrics
- evaluation/sbert/{lang}/{pipeline}/: for sbert metric
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import itertools


# Perturbation categories
CRITICAL_PERTURBATIONS = ['omission', 'expansion_impact', 'alteration']
MINIMAL_PERTURBATIONS = ['spelling', 'synonym', 'expansion_noimpact', 'word_order']

# AskQE metrics to evaluate
ASKQE_METRICS = ['sbert', 'f1', 'em', 'chrf', 'bleu']

# Metric field mappings (evaluation file field name -> our metric name)
METRIC_FIELD_MAP = {
    'sbert': 'sbert_score',
    'f1': 'avg_f1',
    'em': 'avg_em',
    'chrf': 'avg_chrf',
    'bleu': 'avg_bleu'
}


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    return data


def load_perturbation_data(eval_dir, target_lang, pipeline, anscheck_type=None):
    """
    Load all perturbation files for a given pipeline configuration.
    
    The function loads data from two sources:
    - evaluation/string-comparison/{lang}/{pipeline}/: for f1, em, chrf, bleu
    - evaluation/sbert/{lang}/{pipeline}/: for sbert
    
    For anscheck pipelines, files are named {anscheck_type}-{perturbation}.jsonl
    in the anscheck/ subdirectory.
    
    Returns:
        Dictionary: {perturbation: {id: {metric: value}}}
    """
    perturbation_data = {}
    all_perturbations = CRITICAL_PERTURBATIONS + MINIMAL_PERTURBATIONS
    
    for perturbation in all_perturbations:
        # Determine filenames based on pipeline type
        if anscheck_type:
            filename = f"{anscheck_type}-{perturbation}.jsonl"
            string_comp_path = eval_dir / "string-comparison" / target_lang / "anscheck" / filename
            sbert_path = eval_dir / "sbert" / target_lang / "anscheck" / filename
        else:
            filename = f"{perturbation}.jsonl"
            string_comp_path = eval_dir / "string-comparison" / target_lang / pipeline / filename
            sbert_path = eval_dir / "sbert" / target_lang / pipeline / filename
        
        # Check if at least one file exists
        if not string_comp_path.exists() and not sbert_path.exists():
            continue
        
        # Index by id
        indexed_data = {}
        
        # Load string comparison data first
        if string_comp_path.exists():
            data = load_jsonl(string_comp_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    
                    if item_id not in indexed_data:
                        indexed_data[item_id] = {}
                    
                    # Extract string comparison metrics
                    for metric in ['f1', 'em', 'chrf', 'bleu']:
                        field_name = METRIC_FIELD_MAP[metric]
                        if field_name in item:
                            indexed_data[item_id][metric] = item[field_name]
        
        # Load sbert data
        if sbert_path.exists():
            data = load_jsonl(sbert_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    
                    if item_id not in indexed_data:
                        indexed_data[item_id] = {}
                    
                    # Extract sbert metric
                    field_name = METRIC_FIELD_MAP['sbert']
                    if field_name in item:
                        indexed_data[item_id]['sbert'] = item[field_name]
        
        if indexed_data:
            perturbation_data[perturbation] = indexed_data
            print(f"  Loaded {len(indexed_data)} items from {perturbation}")
    
    return perturbation_data


def compute_diff_between(perturbation_data, metric):
    """
    Compute average squared difference between critical and minimal perturbations.
    
    For each sentence ID that appears in all perturbations, compute the squared
    difference between every critical-minimal pair, then return the mean.
    
    Returns:
        tuple: (average_squared_diff, count_of_comparisons)
    """
    # Find available perturbations
    available_critical = [p for p in CRITICAL_PERTURBATIONS if p in perturbation_data]
    available_minimal = [p for p in MINIMAL_PERTURBATIONS if p in perturbation_data]
    
    if not available_critical or not available_minimal:
        return 0.0, 0
    
    # Get common IDs across all available perturbations
    all_perturbations = available_critical + available_minimal
    common_ids = None
    
    for pert in all_perturbations:
        pert_ids = set(perturbation_data[pert].keys())
        if common_ids is None:
            common_ids = pert_ids
        else:
            common_ids = common_ids.intersection(pert_ids)
    
    if not common_ids:
        return 0.0, 0
    
    total_squared_diff = 0.0
    count = 0
    
    # For each common ID, compute squared diff between each critical and minimal pair
    for item_id in common_ids:
        for critical_pert in available_critical:
            if metric not in perturbation_data[critical_pert][item_id]:
                continue
            
            for minimal_pert in available_minimal:
                if metric not in perturbation_data[minimal_pert][item_id]:
                    continue
                
                critical_value = perturbation_data[critical_pert][item_id][metric]
                minimal_value = perturbation_data[minimal_pert][item_id][metric]
                
                squared_diff = (critical_value - minimal_value) ** 2
                total_squared_diff += squared_diff
                count += 1
    
    avg = total_squared_diff / count if count > 0 else 0.0
    return avg, count


def compute_diff_within(perturbation_data, metric):
    """
    Compute average squared difference within critical and within minimal perturbations.
    
    For each category, compute squared differences between all pairs of perturbations
    within that category, then return the overall mean across both categories.
    
    Returns:
        tuple: (average_squared_diff, count_of_comparisons)
    """
    def compute_within_category(perturbations):
        """Compute squared differences within a single category."""
        # Filter to available perturbations
        available = [p for p in perturbations if p in perturbation_data]
        
        if len(available) < 2:
            return 0.0, 0
        
        # Get common IDs for this category
        common_ids = None
        for pert in available:
            pert_ids = set(perturbation_data[pert].keys())
            if common_ids is None:
                common_ids = pert_ids
            else:
                common_ids = common_ids.intersection(pert_ids)
        
        if not common_ids:
            return 0.0, 0
        
        total_squared_diff = 0.0
        count = 0
        
        # For each common ID, compute squared diff between all pairs within category
        for item_id in common_ids:
            for pert1, pert2 in itertools.combinations(available, 2):
                if metric not in perturbation_data[pert1][item_id]:
                    continue
                if metric not in perturbation_data[pert2][item_id]:
                    continue
                
                value1 = perturbation_data[pert1][item_id][metric]
                value2 = perturbation_data[pert2][item_id][metric]
                
                squared_diff = (value1 - value2) ** 2
                total_squared_diff += squared_diff
                count += 1
        
        return total_squared_diff, count
    
    # Compute within critical
    critical_diff, critical_count = compute_within_category(CRITICAL_PERTURBATIONS)
    
    # Compute within minimal
    minimal_diff, minimal_count = compute_within_category(MINIMAL_PERTURBATIONS)
    
    # Return overall average across both categories
    total_diff = critical_diff + minimal_diff
    total_count = critical_count + minimal_count
    avg = total_diff / total_count if total_count > 0 else 0.0
    return avg, total_count


def process_pipeline(eval_dir, target_lang, pipeline, anscheck_type=None):
    """
    Process a single pipeline configuration and return results.
    
    Args:
        eval_dir: Path to evaluation directory
        target_lang: Target language code (e.g., 'en-es')
        pipeline: Pipeline name ('vanilla', 'atomic', or 'anscheck')
        anscheck_type: Anscheck variant ('longformer', 'electra', 'electra-null') if applicable
    
    Returns:
        Dictionary with results or None if no data found
    """
    pipeline_name = f"{pipeline}" + (f" ({anscheck_type})" if anscheck_type else "")
    print(f"\nProcessing pipeline: {pipeline_name}")
    
    # Load perturbation data
    perturbation_data = load_perturbation_data(eval_dir, target_lang, pipeline, anscheck_type)
    
    if not perturbation_data:
        print(f"  No data found for this pipeline configuration")
        return None
    
    results = {
        'pipeline': pipeline,
        'anscheck_type': anscheck_type,
        'metrics': {}
    }
    
    # For each metric, compute DIFF_BETWEEN and DIFF_WITHIN
    for metric in ASKQE_METRICS:
        between_diff, between_count = compute_diff_between(perturbation_data, metric)
        within_diff, within_count = compute_diff_within(perturbation_data, metric)
        
        results['metrics'][metric] = {
            'avg_diff_between': between_diff,
            'diff_between_count': between_count,
            'avg_diff_within': within_diff,
            'diff_within_count': within_count
        }
        
        print(f"  {metric.upper()}:")
        print(f"    AVG_DIFF_BETWEEN: {between_diff:.6f} (n_comparisons: {between_count})")
        print(f"    AVG_DIFF_WITHIN:  {within_diff:.6f} (n_comparisons: {within_count})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute within and between category differences for AskQE metrics'
    )
    parser.add_argument(
        '--eval-dir',
        type=str,
        default='evaluation',
        help='Path to evaluation directory (default: evaluation)'
    )
    parser.add_argument(
        '--target-lang',
        type=str,
        required=True,
        help='Target language code (e.g., en-es, en-fr)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: evaluation/within-between/results_{target_lang}.json)'
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return
    
    all_results = []
    
    # Process vanilla pipeline
    result = process_pipeline(eval_dir, args.target_lang, 'vanilla')
    if result:
        all_results.append(result)
    
    # Process atomic pipeline
    result = process_pipeline(eval_dir, args.target_lang, 'atomic')
    if result:
        all_results.append(result)
    
    # Process anscheck pipelines
    for anscheck_type in ['longformer', 'electra', 'electra-null']:
        result = process_pipeline(eval_dir, args.target_lang, 'anscheck', anscheck_type)
        if result:
            all_results.append(result)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('evaluation/within-between') / f'results_{args.target_lang}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_data = {
        'target_lang': args.target_lang,
        'critical_perturbations': CRITICAL_PERTURBATIONS,
        'minimal_perturbations': MINIMAL_PERTURBATIONS,
        'results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

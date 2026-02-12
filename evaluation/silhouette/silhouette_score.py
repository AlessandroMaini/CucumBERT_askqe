import json
import argparse
from pathlib import Path

from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Perturbation categories
CRITICAL_PERTURBATIONS = ['omission', 'alteration']
MINIMAL_PERTURBATIONS = ['synonym', 'expansion_noimpact']

# AskQE metrics (pipeline-dependent)
ASKQE_METRICS = ['sbert', 'f1', 'em', 'chrf', 'bleu']

# Standard MT metrics (pipeline-independent)
MT_METRICS = ['xcomet', 'bertscore']

# Metric field mappings (evaluation file field name -> our metric name)
METRIC_FIELD_MAP = {
    'sbert': 'sbert_score',
    'f1': 'avg_f1',
    'em': 'avg_em',
    'chrf': 'avg_chrf',
    'bleu': 'avg_bleu',
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
    Load pipeline-dependent AskQE perturbation files.

    Sources:
      evaluation/string-comparison/{lang}/{pipeline}/  (f1, em, chrf, bleu)
      evaluation/sbert/{lang}/{pipeline}/               (sbert)

    Returns:
        Dictionary: {perturbation: {id: {metric: value}}}
    """
    perturbation_data = {}
    all_perturbations = CRITICAL_PERTURBATIONS + MINIMAL_PERTURBATIONS

    for perturbation in all_perturbations:
        if anscheck_type:
            filename = f"{anscheck_type}-{perturbation}.jsonl"
            string_comp_path = eval_dir / "string-comparison" / target_lang / "anscheck" / filename
            sbert_path = eval_dir / "sbert" / target_lang / "anscheck" / filename
        else:
            filename = f"{perturbation}.jsonl"
            string_comp_path = eval_dir / "string-comparison" / target_lang / pipeline / filename
            sbert_path = eval_dir / "sbert" / target_lang / pipeline / filename

        if not string_comp_path.exists() and not sbert_path.exists():
            continue

        indexed_data: dict = {}

        if string_comp_path.exists():
            data = load_jsonl(string_comp_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    indexed_data.setdefault(item_id, {})
                    for metric in ['f1', 'em', 'chrf', 'bleu']:
                        field = METRIC_FIELD_MAP[metric]
                        if field in item:
                            indexed_data[item_id][metric] = item[field]

        if sbert_path.exists():
            data = load_jsonl(sbert_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    indexed_data.setdefault(item_id, {})
                    field = METRIC_FIELD_MAP['sbert']
                    if field in item:
                        indexed_data[item_id]['sbert'] = item[field]

        if indexed_data:
            perturbation_data[perturbation] = indexed_data
            print(f"  Loaded {len(indexed_data)} items from {perturbation}")

    return perturbation_data


def load_mt_data(eval_dir, target_lang):
    """
    Load pipeline-independent standard MT metric files.

    Sources:
      evaluation/xcomet/{lang}/{perturbation}.jsonl                (xcomet)
      evaluation/bt-score/{lang}/bt-{perturbation}_bertscore.jsonl (bertscore)

    Returns:
        Dictionary: {perturbation: {id: {metric: value}}}
    """
    perturbation_data = {}
    all_perturbations = CRITICAL_PERTURBATIONS + MINIMAL_PERTURBATIONS

    for perturbation in all_perturbations:
        xcomet_path = eval_dir / "xcomet" / target_lang / f"{perturbation}.jsonl"
        bertscore_path = eval_dir / "bt-score" / target_lang / f"bt-{perturbation}_bertscore.jsonl"

        if not xcomet_path.exists() and not bertscore_path.exists():
            continue

        indexed_data: dict = {}

        if xcomet_path.exists():
            data = load_jsonl(xcomet_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    indexed_data.setdefault(item_id, {})
                    annotation = item.get('xcomet_annotation', {})
                    if 'segment_score' in annotation:
                        indexed_data[item_id]['xcomet'] = annotation['segment_score']

        if bertscore_path.exists():
            data = load_jsonl(bertscore_path)
            if data:
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    indexed_data.setdefault(item_id, {})
                    if 'bertscore_f1' in item:
                        indexed_data[item_id]['bertscore'] = item['bertscore_f1']

        if indexed_data:
            perturbation_data[perturbation] = indexed_data
            print(f"  Loaded {len(indexed_data)} items from {perturbation}")

    return perturbation_data


def collect_scores_1d(perturbation_data, metric):
    """
    Collect individual metric scores with category labels for 1D silhouette analysis.

    Each perturbation score becomes one 1D sample.  Critical perturbations
    are labelled 0, minimal perturbations are labelled 1.

    Returns:
        tuple: (scores_array shape (N,1), labels_array shape (N,))
               or (None, None) when fewer than 2 labels are present.
    """
    CATEGORY_MAP = {
        **{p: 0 for p in CRITICAL_PERTURBATIONS},
        **{p: 1 for p in MINIMAL_PERTURBATIONS},
    }

    scores = []
    labels = []

    for pert, items in perturbation_data.items():
        label = CATEGORY_MAP.get(pert)
        if label is None:
            continue
        for item_id, metrics in items.items():
            if metric in metrics:
                scores.append(metrics[metric])
                labels.append(label)

    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int)

    # Silhouette requires at least 2 distinct labels with >=1 sample each
    if len(np.unique(labels)) < 2 or len(scores) < 2:
        return None, None

    return scores, labels


def plot_kde(perturbation_data, metric, pipeline_name, target_lang, output_dir):
    """
    Plot KDE distributions of a metric for critical vs minimal perturbations.

    Produces a single figure with overlapping KDE curves coloured by category
    and saves it to *output_dir*.

    Args:
        perturbation_data: dict {perturbation: {id: {metric: value}}}
        metric: metric name (e.g. 'f1', 'sbert')
        pipeline_name: label used in plot title and filename
        target_lang: language pair string (e.g. 'en-es')
        output_dir: Path where the PNG will be saved
    """
    CATEGORY_NAMES = {0: 'critical', 1: 'minimal'}
    scores, labels = collect_scores_1d(perturbation_data, metric)

    if scores is None:
        print(f"    Skipping KDE plot for {metric} — insufficient data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for label_id, name in CATEGORY_NAMES.items():
        mask = labels == label_id
        if mask.sum() == 0:
            continue
        sns.kdeplot(scores[mask].ravel(), ax=ax, label=name, fill=True, alpha=0.35)

    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Density')
    ax.set_title(f'{metric.upper()} — {pipeline_name} ({target_lang})')
    ax.legend(title='Category')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f'kde_{metric}_{pipeline_name}_{target_lang}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    KDE plot saved to {fname}")

def process_pipeline(eval_dir, target_lang, pipeline, anscheck_type=None,
                     do_plot=False):
    """
    Process a single pipeline configuration and return results.
    
    Args:
        eval_dir: Path to evaluation directory
        target_lang: Target language code (e.g., 'en-es')
        pipeline: Pipeline name ('vanilla', 'atomic', or 'anscheck')
        anscheck_type: Anscheck variant ('longformer', 'electra') if applicable
        do_plot: Whether to generate KDE plots
    
    Returns:
        Dictionary with results or None if no data found
    """
    pipeline_name = f"{pipeline}" + (f"_{anscheck_type}" if anscheck_type else "")
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

    plot_dir = Path(__file__).resolve().parent / 'plots'

    for metric in ASKQE_METRICS:
        scores, labels = collect_scores_1d(perturbation_data, metric)

        silhouette = None
        n_samples = 0
        if scores is not None:
            silhouette = silhouette_score(scores, labels)
            n_samples = scores.shape[0]

        results['metrics'][metric] = {
            'silhouette_score': silhouette,
            'n_samples': n_samples,
        }

        sil_str = f"{silhouette:.6f}" if silhouette is not None else "N/A"
        print(f"  {metric.upper():<12} silhouette={sil_str}  (n={n_samples})")

        if do_plot:
            plot_kde(perturbation_data, metric, pipeline_name, target_lang,
                     plot_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute 1-D silhouette scores for AskQE and standard MT metrics'
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
        help='Output file path (default: evaluation/silhouette/results_{target_lang}.json)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help='Generate KDE plots for each metric/pipeline'
    )
    
    args = parser.parse_args()
    
    # Get workspace root (2 levels up from this script in evaluation/silhouette/)
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent
    
    eval_dir = workspace_root / 'evaluation'
    
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return
    
    all_results = []

    # Process vanilla pipeline
    result = process_pipeline(eval_dir, args.target_lang, 'vanilla',
                              do_plot=args.plot)
    if result:
        all_results.append(result)

    # Process atomic pipeline
    result = process_pipeline(eval_dir, args.target_lang, 'atomic',
                              do_plot=args.plot)
    if result:
        all_results.append(result)

    # Process factcoverage pipeline
    result = process_pipeline(eval_dir, args.target_lang, 'factcoverage',
                              do_plot=args.plot)
    if result:
        all_results.append(result)

    # --- Compute MT metrics once (pipeline-independent) ---
    print(f"\nComputing pipeline-independent MT metrics...")
    mt_data = load_mt_data(eval_dir, args.target_lang)
    mt_results = {}
    plot_dir = Path(__file__).resolve().parent / 'plots'

    for metric in MT_METRICS:
        scores, labels = collect_scores_1d(mt_data, metric)
        silhouette = None
        n_samples = 0
        if scores is not None:
            silhouette = silhouette_score(scores, labels)
            n_samples = scores.shape[0]
        mt_results[metric] = {
            'silhouette_score': silhouette,
            'n_samples': n_samples,
        }
        sil_str = f"{silhouette:.6f}" if silhouette is not None else "N/A"
        print(f"  {metric.upper():<12} silhouette={sil_str}  (n={n_samples})")

        if args.plot:
            plot_kde(mt_data, metric, 'mt', args.target_lang, plot_dir)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / f'results_{args.target_lang}.json'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    output_data = {
        'target_lang': args.target_lang,
        'critical_perturbations': CRITICAL_PERTURBATIONS,
        'minimal_perturbations': MINIMAL_PERTURBATIONS,
        'askqe_metrics': ASKQE_METRICS,
        'mt_metrics': MT_METRICS,
        'mt_results': mt_results,
        'results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

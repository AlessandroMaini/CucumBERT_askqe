"""
Compute instance-based Pearson correlations between all combinations of
AskQE metrics, standard MT QE metrics, and pipelines for a single dataset.

For each (AskQE metric+pipeline, standard metric) pair, the correlation is
computed over individual sentences (instance-based).  The output JSON also
includes per-perturbation average scores but does NOT store per-sentence data.

Produces one JSON file per dataset: e.g.  results/en-es.json

Usage:
    python compute_correlation.py --dataset en-es
    python compute_correlation.py --dataset en-fr
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from itertools import product

# ── Configuration ────────────────────────────────────────────────────────────

PERTURBATIONS = ["synonym", "alteration", "omission", "expansion_noimpact"]
DATASETS = ["en-es", "en-fr"]
PIPELINES_ASKQE = [
    {"name": "vanilla"},
    {"name": "atomic"},
    {"name": "anscheck", "anscheck_type": "longformer"},
    {"name": "anscheck", "anscheck_type": "electra"},
]

STANDARD_METRICS = {
    "xcomet": {
        "base_path": "evaluation/xcomet",
        "file_pattern": lambda dataset, pert: f"{dataset}/{pert}.jsonl",
        "field_extractor": lambda d: d.get("xcomet_annotation", {}).get("segment_score"),
    },
    "bt_score": {
        "base_path": "evaluation/bt-score",
        "file_pattern": lambda dataset, pert: f"{dataset}/bt-{pert}_bertscore.jsonl",
        "field_extractor": lambda d: d.get("bertscore_f1"),
    },
}

ASKQE_METRICS = {
    "f1": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda d: d.get("avg_f1") if d.get("avg_f1") is not None else (
            np.mean([s["f1"] for s in d.get("scores", [])]) if d.get("scores") else None
        ),
    },
    "em": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda d: d.get("avg_em") if d.get("avg_em") is not None else (
            np.mean([1.0 if s["em"] else 0.0 for s in d.get("scores", [])]) if d.get("scores") else None
        ),
    },
    "bleu": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda d: d.get("avg_bleu") if d.get("avg_bleu") is not None else (
            np.mean([s["bleu"] for s in d.get("scores", [])]) if d.get("scores") else None
        ),
    },
    "chrf": {
        "base_path": "evaluation/string-comparison",
        "field_extractor": lambda d: d.get("avg_chrf") if d.get("avg_chrf") is not None else (
            np.mean([s["chrf"] for s in d.get("scores", [])]) if d.get("scores") else None
        ),
    },
    "sbert": {
        "base_path": "evaluation/sbert",
        "field_extractor": lambda d: d.get("sbert_score"),
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_jsonl(path):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pipeline_label(pcfg):
    label = pcfg["name"]
    if pcfg.get("anscheck_type"):
        label += f"-{pcfg['anscheck_type']}"
    return label


# ── Data loading ─────────────────────────────────────────────────────────────

def load_standard_metric(metric_name, dataset, workspace_root):
    """
    Load per-instance scores for a standard metric on one dataset.

    Returns:
        all_scores : dict  {composite_key: score}   (instance-level)
        pert_avgs  : dict  {perturbation: avg_score}
    """
    cfg = STANDARD_METRICS[metric_name]
    base = workspace_root / cfg["base_path"]

    all_scores = {}
    pert_accum = {p: [] for p in PERTURBATIONS}

    for pert in PERTURBATIONS:
        fp = base / cfg["file_pattern"](dataset, pert)
        if not fp.exists():
            print(f"  [skip] {fp}")
            continue
        for record in read_jsonl(fp):
            rid = record.get("id")
            if not rid:
                continue
            val = cfg["field_extractor"](record)
            if val is not None:
                val = float(val)
                all_scores[f"{rid}_{pert}"] = val
                pert_accum[pert].append(val)

    pert_avgs = {
        p: round(float(np.mean(v)), 6) if v else None
        for p, v in pert_accum.items()
    }
    return all_scores, pert_avgs


def load_askqe_metric(metric_name, pipeline_cfg, dataset, workspace_root):
    """
    Load per-instance scores for an AskQE metric + pipeline on one dataset.

    Returns:
        all_scores : dict  {composite_key: score}
        pert_avgs  : dict  {perturbation: avg_score}
    """
    cfg = ASKQE_METRICS[metric_name]
    base = workspace_root / cfg["base_path"]
    pipeline = pipeline_cfg["name"]
    anscheck_type = pipeline_cfg.get("anscheck_type")

    all_scores = {}
    pert_accum = {p: [] for p in PERTURBATIONS}

    for pert in PERTURBATIONS:
        if pipeline == "anscheck":
            filename = f"{anscheck_type}-{pert}.jsonl"
        else:
            filename = f"{pert}.jsonl"
        fp = base / dataset / pipeline / filename
        if not fp.exists():
            print(f"  [skip] {fp}")
            continue
        for record in read_jsonl(fp):
            rid = record.get("id")
            if not rid:
                continue
            val = cfg["field_extractor"](record)
            if val is not None:
                val = float(val)
                all_scores[f"{rid}_{pert}"] = val
                pert_accum[pert].append(val)

    pert_avgs = {
        p: round(float(np.mean(v)), 6) if v else None
        for p, v in pert_accum.items()
    }
    return all_scores, pert_avgs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute instance-based Pearson correlations for all metric "
                    "combinations on a single dataset."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASETS,
        help="Language dataset (en-es or en-fr)"
    )
    args = parser.parse_args()
    dataset = args.dataset

    workspace_root = Path(__file__).resolve().parent.parent.parent

    # ── 1. Load all standard metrics ────────────────────────────────────────
    std_data = {}      # {metric_name: {key: score}}
    std_avgs = {}      # {metric_name: {pert: avg}}

    for sm in STANDARD_METRICS:
        print(f"Loading standard metric: {sm}")
        scores, avgs = load_standard_metric(sm, dataset, workspace_root)
        std_data[sm] = scores
        std_avgs[sm] = avgs
        print(f"  {len(scores)} instances loaded")

    # ── 2. Load all AskQE metrics × pipelines ──────────────────────────────
    askqe_data = {}    # {label: {key: score}}
    askqe_avgs = {}    # {label: {pert: avg}}

    for am, pcfg in product(ASKQE_METRICS, PIPELINES_ASKQE):
        label = f"{am}-{pipeline_label(pcfg)}"
        print(f"Loading AskQE metric: {label}")
        scores, avgs = load_askqe_metric(am, pcfg, dataset, workspace_root)
        askqe_data[label] = scores
        askqe_avgs[label] = avgs
        print(f"  {len(scores)} instances loaded")

    # ── 3. Compute all correlations ─────────────────────────────────────────
    correlations = []

    for ak, sm in product(askqe_data, std_data):
        common_keys = sorted(set(askqe_data[ak]) & set(std_data[sm]))
        n = len(common_keys)

        if n < 3:
            print(f"  [warn] {ak} vs {sm}: only {n} common instances, skipping")
            correlations.append({
                "askqe_metric": ak,
                "standard_metric": sm,
                "n_instances": n,
                "pearson_r": None,
                "p_value": None,
                "note": "Not enough common instances",
            })
            continue

        vals_a = [askqe_data[ak][k] for k in common_keys]
        vals_s = [std_data[sm][k] for k in common_keys]
        r, p = pearsonr(vals_a, vals_s)

        correlations.append({
            "askqe_metric": ak,
            "standard_metric": sm,
            "n_instances": n,
            "pearson_r": round(float(r), 6),
            "p_value": float(p),
        })

    # ── 4. Build output JSON ────────────────────────────────────────────────
    results = {
        "dataset": dataset,
        "perturbation_types": PERTURBATIONS,
        "description": (
            "Instance-based Pearson correlations between AskQE and standard MT QE metrics. "
            "Correlations are computed over individual sentences across all perturbation types. "
            "Per-perturbation averages are included for reference."
        ),
        "standard_metric_averages": std_avgs,
        "askqe_metric_averages": askqe_avgs,
        "correlations": correlations,
    }

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"results_{dataset}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 5. Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset}")
    print(f"{'='*70}")

    print("\nPer-perturbation averages (standard):")
    for sm, avgs in std_avgs.items():
        vals = "  ".join(f"{p}: {v:.4f}" if v else f"{p}: N/A" for p, v in avgs.items())
        print(f"  {sm:>12s} | {vals}")

    print("\nPer-perturbation averages (AskQE):")
    for ak, avgs in askqe_avgs.items():
        vals = "  ".join(f"{p}: {v:.4f}" if v else f"{p}: N/A" for p, v in avgs.items())
        print(f"  {ak:>20s} | {vals}")

    print(f"\n{'='*70}")
    print("INSTANCE-BASED PEARSON CORRELATIONS")
    print(f"{'='*70}")
    print(f"  {'AskQE':>20s}  {'Standard':>12s}  {'r':>8s}  {'p':>12s}  {'n':>6s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*6}")
    for c in correlations:
        r_str = f"{c['pearson_r']:.4f}" if c["pearson_r"] is not None else "N/A"
        p_str = f"{c['p_value']:.4e}" if c["p_value"] is not None else "N/A"
        print(f"  {c['askqe_metric']:>20s}  {c['standard_metric']:>12s}  {r_str:>8s}  {p_str:>12s}  {c['n_instances']:>6d}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

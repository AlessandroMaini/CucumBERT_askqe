"""
Plot heatmaps of instance-based Pearson correlations from result JSON files.

Shows pipelines side by side: one sub-heatmap per pipeline, with rows = base
AskQE metrics (F1, EM, BLEU, CHRF, SBERT) and columns = standard metrics.

Usage:
    python plot_instance_correlation.py                          # plots all results_*.json found
    python plot_instance_correlation.py --files results_en-es.json results_en-fr.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict


STANDARD_LABELS = {
    "xcomet": "xCOMET",
    "bt_score": "BERTScore",
}

PIPELINE_LABELS = {
    "vanilla": "Vanilla",
    "atomic": "Atomic",
    "anscheck-longformer": "AnsCheck-LF",
    "anscheck-electra": "AnsCheck-EL",
}

BASE_METRIC_ORDER = ["F1", "EM", "BLEU", "CHRF", "SBERT"]


def parse_askqe_key(key):
    """'f1-vanilla' → ('F1', 'vanilla')"""
    parts = key.split("-", 1)
    return parts[0].upper(), parts[1] if len(parts) > 1 else ""


def plot_heatmap(results, output_path):
    correlations = results["correlations"]
    dataset = results["dataset"]

    # Organise correlations into {pipeline: {(base_metric, std_metric): r}}
    pipelines = OrderedDict()
    std_keys_set = OrderedDict()
    for c in correlations:
        base, pipe = parse_askqe_key(c["askqe_metric"])
        sm = c["standard_metric"]
        std_keys_set[sm] = True
        pipelines.setdefault(pipe, {})
        pipelines[pipe][(base, sm)] = c["pearson_r"]

    std_keys = list(std_keys_set.keys())
    base_metrics = [m for m in BASE_METRIC_ORDER if any(m in k for pipe in pipelines.values() for k in pipe)]
    n_pipes = len(pipelines)

    # Global colour range across all pipelines
    all_r = [v for pipe in pipelines.values() for v in pipe.values() if v is not None]
    vmin = min(all_r) - 0.02 if all_r else 0
    vmax = max(all_r) + 0.02 if all_r else 1

    fig, axes = plt.subplots(1, n_pipes, figsize=(3.6 * n_pipes + 1.2, 4.2),
                              sharey=True, constrained_layout=True)
    if n_pipes == 1:
        axes = [axes]

    ims = []
    for idx, (pipe, data) in enumerate(pipelines.items()):
        ax = axes[idx]
        mat = np.full((len(base_metrics), len(std_keys)), np.nan)
        for i, bm in enumerate(base_metrics):
            for j, sm in enumerate(std_keys):
                val = data.get((bm, sm))
                if val is not None:
                    mat[i, j] = val

        im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ims.append(im)

        # Annotate cells
        for i in range(len(base_metrics)):
            for j in range(len(std_keys)):
                r = mat[i, j]
                txt = f"{r:.3f}" if not np.isnan(r) else "N/A"
                tc = "white" if (not np.isnan(r) and r < (vmin + vmax) / 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        fontweight="bold", color=tc)

        ax.set_xticks(range(len(std_keys)))
        ax.set_xticklabels([STANDARD_LABELS.get(k, k) for k in std_keys], fontsize=10)
        ax.set_title(PIPELINE_LABELS.get(pipe, pipe.capitalize()), fontsize=12, pad=14)

        if idx == 0:
            ax.set_yticks(range(len(base_metrics)))
            ax.set_yticklabels(base_metrics, fontsize=11)
        else:
            ax.tick_params(left=False)

    # Shared colour bar
    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.75, pad=0.03)
    cbar.set_label("Pearson r", fontsize=11)

    fig.suptitle(f"Instance-Based Pearson Correlations — {dataset}",
                 fontsize=13, y=1.06)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot heatmaps of instance-based correlations.")
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="JSON result files. If omitted, all results_*.json in the script directory are used.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots (default: plots/ next to the script).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        json_files = sorted(script_dir.glob("results_*.json"))

    if not json_files:
        print("No result files found. Run compute_correlation.py first.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            results = json.load(f)
        dataset = results["dataset"]
        out_path = output_dir / f"heatmap_instance_{dataset}.png"
        plot_heatmap(results, out_path)

    print(f"\nAll heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
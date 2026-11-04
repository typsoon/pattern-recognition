"""Aggregate final experiment results and generate the shared comparison figure."""

import argparse
import csv
import json
import math
import statistics

import matplotlib.pyplot as plt

from experiment_utils import add_output_arguments, prepare_output_dir


HISTORY_FILES = {
    "Default DINO": "dino_default_c1_s1_seed0_history.json",
    "No centering (collapsed baseline)": "dino_no_center_c0_s1_seed0_history.json",
    "Target standardization (Person 2)": "standardize_best_seed0_history.json",
    "Mutual information (Person 3)": "mutual_info_best_seed0_history.json",
}


def _read_final_results(results_path):
    if not results_path.is_file():
        raise FileNotFoundError(f"Experiment results not found: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No experiment rows found in {results_path}")

    required = {"method", "seed", "knn_eval", "timestamp", "note"}
    if not required.issubset(rows[0]):
        missing = ", ".join(sorted(required.difference(rows[0])))
        raise ValueError(f"results.csv is missing required columns: {missing}")

    final_rows = [
        row
        for row in rows
        if row["note"] == "final_eval" and row["knn_eval"].strip()
    ]
    if not final_rows:
        raise ValueError("results.csv contains no completed final evaluations.")

    # Keep the latest completed result if a person reran a final seed.
    latest = {}
    for row in final_rows:
        key = (row["method"], int(row["seed"]))
        if key not in latest or row["timestamp"] > latest[key]["timestamp"]:
            latest[key] = row
    return list(latest.values())


def _summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(float(row["knn_eval"]))
    return {
        method: {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
        for method, values in sorted(grouped.items())
    }


def _print_latex_table(summary):
    print("\n=== FINAL TEST ACCURACY SUMMARY (knn_eval) ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"Method & Seeds & Mean accuracy (\%) & Std. dev. (\%) \\")
    print(r"\hline")
    for method, values in summary.items():
        mean = values["mean"] * 100
        std = values["std"]
        std_text = "--" if std is None else f"{std * 100:.2f}"
        print(
            f"{method} & {values['count']} & {mean:.2f} & {std_text} \\\\"
        )
    print(r"\end{tabular}")


def _plot_histories(output_dir):
    checkpoint_dir = output_dir / "checkpoints"
    histories = {}
    missing = []
    for label, filename in HISTORY_FILES.items():
        path = checkpoint_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(encoding="utf-8") as stream:
            histories[label] = json.load(stream)
        if not histories[label]:
            raise ValueError(f"Training history is empty: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing histories required for the comparison figure:\n"
            + "\n".join(missing)
        )

    panels = (
        ("entropy", "Teacher sample entropy $H(P_t)$"),
        ("batch_entropy", "Batch marginal entropy $H(\\bar{P}_t)$"),
        ("kl", "Teacher–student KL divergence"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, (key, title) in zip(axes, panels):
        for label, history in histories.items():
            axis.plot(
                [entry["step"] for entry in history],
                [entry[key] for entry in history],
                label=label,
                linewidth=2,
            )
        if key in {"entropy", "batch_entropy"}:
            axis.axhline(
                math.log(1024),
                color="gray",
                linestyle="--",
                alpha=0.8,
                label="$\\log(1024)$",
            )
        axis.set(title=title, xlabel="Training step")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Nats")
    axes[0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure_path = output_dir / "figures" / "master_method_comparison.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def run_comparison(output_dir):
    rows = _read_final_results(output_dir / "results.csv")
    summary = _summarize(rows)
    _print_latex_table(summary)

    summary_path = output_dir / "final_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("method", "count", "mean", "std")
        )
        writer.writeheader()
        for method, values in summary.items():
            writer.writerow({"method": method, **values})

    figure_path = _plot_histories(output_dir)
    print(f"Saved numeric summary: {summary_path}")
    print(f"Saved comparison figure: {figure_path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    output_dir = prepare_output_dir(
        args.output_dir, mount_drive=not args.no_drive_mount
    )
    run_comparison(output_dir)


if __name__ == "__main__":
    main()

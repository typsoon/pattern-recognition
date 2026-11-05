"""Part 2, Person 3: prevent collapse with a mutual-information regulariser.

Without centering and sharpening (teacher and student share one temperature) DINO
collapses in one of two ways, and in both the code no longer depends on the image:

    every image on the same code     ->  H(P_bar) = 0,       mean H(P_s) = 0
    uniform output for every image   ->  H(P_bar) = log K,   mean H(P_s) = log K

P_s are the student's distributions and P_bar is their average over the batch.
We add two terms to the DINO cross-entropy:

    - lambda_marg * H(P_bar)        use many codes across the batch  -> blocks "same code"
    + lambda_cond * mean H(P_s)     be confident on every image      -> blocks "uniform"

With lambda_marg = lambda_cond this is exactly -lambda * I(image; code), the mutual
information between an image and its code, which is 0 in both collapses.

Run in Colab from the lab4 directory, after the setup cells:   %run fix_mutual_info.py
1. trains the CONFIGS on seed 0 and compares them with val_knn only,
2. trains the best configuration on seeds 1 and 2,
3. reports knn_eval for seeds 0, 1 and 2 (seed 0 is evaluated only now, after selection),
4. saves weights, histories, figures and results.csv rows in the shared Drive folder.
Finished runs are loaded from Drive, so after a disconnect just run the cell again.
"""

import argparse
import csv
import json
import statistics

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common import device, normalize
from evaluation import knn_eval, val_knn
from experiment_utils import (
    add_output_arguments,
    append_result,
    load_lab_data,
    prepare_output_dir,
    save_run_artifacts,
    set_all_seeds,
)
from network import DINONetwork
from plots import plot_variants
from training import train_dino

METHOD = "mutual_info"  # The method name used in results.csv and the saved artifacts.
# (lambda_marg, lambda_cond, tau, K): the configurations tuned on seed 0 with val_knn.
CONFIGS = [
    # the six configurations agreed in the group
    (1.5, 0.1, 0.07, 1024),
    (2.0, 0.2, 0.07, 1024),
    (2.5, 0.3, 0.07, 1024),
    (2.0, 0.2, 0.05, 1024),
    (2.0, 0.2, 0.10, 1024),
    (2.0, 0.2, 0.07, 512),
    # equal weights (exactly -lambda * I). An earlier exploration with equal weights gave
    # val_knn 0.458 / 0.379 / 0.308 for lambda = 0.5 / 1 / 2 (tau = 0.05): smaller was better,
    # and 0.5 was the smallest value tried, so we also try 0.25.
    (0.5, 0.5, 0.05, 1024),
    (0.25, 0.25, 0.05, 1024),
]
TUNING_SEED = 0  # Every configuration is trained once, on this seed, to choose the best one.
FINAL_SEEDS = (0, 1, 2)  # The best configuration is reported as a mean over these seeds.


def entropy(p):
    # Shannon entropy in nats of each distribution along the last axis.
    # Sum -p*log(p) over the K codes; clamp avoids log(0) for codes with zero probability.
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def mutual_info_loss(logits, temperature, lambda_marg, lambda_cond):
    # logits - (2B, K) student logits of both views, concatenated
    # returns: the regulariser added to the DINO loss, a scalar with gradient to the student
    # The same temperature as in the DINO loss turns logits into one distribution per image.
    p = F.softmax(logits / temperature, dim=-1)
    # Average the distributions over the batch, then take one entropy of that average:
    # it is high when the batch uses many codes, 0 when every image uses the same code.
    marginal = entropy(p.mean(0))
    # Take one entropy per image, then average: it is low when each image's code is
    # confident, log K when every output is uniform.
    conditional = entropy(p).mean()
    # Reward a high marginal entropy, penalise a high conditional entropy.
    return -lambda_marg * marginal + lambda_cond * conditional


def run_name(config, seed):
    # A unique file name per configuration and seed, e.g. mi_marg2.0_cond0.2_tau0.07_K1024_seed0.
    lambda_marg, lambda_cond, tau, K = config
    return f"mi_marg{lambda_marg}_cond{lambda_cond}_tau{tau}_K{K}_seed{seed}"


def train(config, seed, train_images, train_labels):
    # DINO without centering and sharpening (tau_t = tau_s = tau), plus the regulariser.
    lambda_marg, lambda_cond, tau, K = config
    set_all_seeds(seed)  # Reproduce initialization, batch order and augmentations.

    def regulariser(s1, s2):
        # train_dino calls this with the student logits of the two views at every step.
        # Concatenating them estimates the batch-average code from 2B distributions.
        return mutual_info_loss(torch.cat([s1, s2]), tau, lambda_marg, lambda_cond)

    # center=False keeps the center at zero; sharpen=False sets tau_t = tau_s = tau.
    return train_dino(
        train_images, train_labels, center=False, sharpen=False,
        seed=seed, K=K, tau_s=tau, extra_loss=regulariser,
    )


def load_run(output_dir, name, K):
    # returns: (run in train_dino's format, val_knn) of a run saved by save_run_artifacts,
    #          or (None, None). The summary is saved last, so it exists only for complete runs.
    folder = output_dir / "checkpoints"
    if not (folder / f"{name}_summary.json").exists():
        return None, None
    saved = torch.load(folder / f"{name}.pt", map_location=device)  # Move saved tensors to this device.
    # Rebuild the same dictionary train_dino returns, so saved and fresh runs are used alike.
    run = {
        "center": saved["center"].to(device),  # Always zero here, since centering is off.
        "tau_t": saved["tau_t"],  # The shared temperature.
        "history": json.loads((folder / f"{name}_history.json").read_text()),  # Logged curves.
    }
    for role in ("student", "teacher"):
        # Recreate the network with the saved number of codes, then load its weights.
        run[role] = DINONetwork(K).to(device)
        run[role].load_state_dict(saved[f"{role}_state"])
    run["teacher"].requires_grad_(False)  # The teacher is never trained by gradients.
    val_acc = json.loads((folder / f"{name}_summary.json").read_text())["val_knn"]
    return run, val_acc


def train_or_load(output_dir, config, seed, train_images, train_labels):
    # returns: (run, val_knn); trains and saves only when the run is not on Drive yet
    name = run_name(config, seed)
    run, val_acc = load_run(output_dir, name, config[3])  # config[3] is K.
    if run is not None:  # Reuse a completed run instead of spending time training again.
        print(f"loaded  {name}: val_knn {val_acc:.4f}")
        return run, val_acc
    run = train(config, seed, train_images, train_labels)
    # Score with the validation split only; the test set is not used for any choice.
    val_acc = val_knn(run["student"].backbone, train_images, train_labels)
    lambda_marg, lambda_cond, tau, K = config
    # Save weights, history and configuration, and add a results.csv row without knn_eval.
    save_run_artifacts(
        output_dir, artifact_name=name, method=METHOD, seed=seed, result=run,
        configuration=dict(lambda_marg=lambda_marg, lambda_cond=lambda_cond, tau=tau, K=K),
        val_acc=val_acc, test_acc=None, note=f"trained {name}",
    )
    print(f"trained {name}: val_knn {val_acc:.4f}")
    return run, val_acc


def reported_results(output_dir):
    # returns: {note: knn_eval} of the final rows already in results.csv,
    #          so running the script again does not evaluate or write them twice
    path = output_dir / "results.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["note"]: float(row["knn_eval"])
            for row in csv.DictReader(stream)
            # Only this method's final rows; training rows have no knn_eval.
            if row["method"] == METHOD and row["note"].startswith("final")
        }


def main(argv=None):
    # Shared options from experiment_utils: --output-dir and --no-drive-mount.
    parser = argparse.ArgumentParser(description="Mutual-information fix (Task 4, Part 2).")
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    # Mount Drive and create the checkpoints/ and figures/ folders.
    output_dir = prepare_output_dir(args.output_dir, mount_drive=not args.no_drive_mount)
    train_images, train_labels, test_images, test_labels = load_lab_data()

    # 1. Tuning on seed 0: choose the configuration with val_knn only.
    tuning = {}  # config -> (run, val_knn)
    for config in CONFIGS:
        tuning[config] = train_or_load(output_dir, config, TUNING_SEED, train_images, train_labels)
    best = max(CONFIGS, key=lambda config: tuning[config][1])  # The highest val_knn wins.
    print("\nval_knn on seed 0 (lambda_marg, lambda_cond, tau, K):")
    for config in sorted(CONFIGS, key=lambda config: -tuning[config][1]):  # Best first.
        print(f"  {config}: {tuning[config][1]:.4f}" + ("   <- best" if config == best else ""))

    # 2. Final numbers: the best configuration on every seed, now with knn_eval.
    reported = reported_results(output_dir)
    test_accs = []
    for seed in FINAL_SEEDS:
        # Seed 0 is loaded from the tuning step; seeds 1 and 2 are trained here.
        run, val_acc = train_or_load(output_dir, best, seed, train_images, train_labels)
        note = f"final {run_name(best, seed)}"
        if note not in reported:  # Evaluate on the test set once per seed, after selection.
            reported[note] = knn_eval(
                run["student"].backbone, train_images, train_labels, test_images, test_labels
            )
            append_result(output_dir, method=METHOD, seed=seed, val_acc=val_acc,
                          test_acc=reported[note], note=note)
        test_accs.append(reported[note])
        print(f"seed {seed}: knn_eval {reported[note]:.4f}")
    # Report the mean and the standard deviation over seeds, in percent.
    print(f"\nbest {best}: knn_eval {100 * statistics.mean(test_accs):.2f} "
          f"± {100 * statistics.stdev(test_accs):.2f} % over seeds {FINAL_SEEDS}")

    # 3. Figures of the tuning runs: teacher targets on two test images and training curves.
    # The same plots as in Part 1; here the center is zero and tau_t equals tau_s.
    labelled = {f"λ_marg={c[0]}, λ_cond={c[1]}, τ={c[2]}, K={c[3]}": tuning[c][0] for c in CONFIGS}
    stages_fig, curves_fig = plot_variants(labelled, normalize(test_images[:2]))
    stages_fig.savefig(output_dir / "figures" / "mi_teacher_softmax_stages.png", dpi=150, bbox_inches="tight")
    curves_fig.savefig(output_dir / "figures" / "mi_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()  # Display both figures in the notebook.


if __name__ == "__main__":  # `%run fix_mutual_info.py` runs this block.
    main()

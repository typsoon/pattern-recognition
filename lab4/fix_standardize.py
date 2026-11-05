"""Person 2: DINO with target-side batch moment standardization."""

import argparse

import torch
from torch.nn import functional as F

from evaluation import knn_eval, val_knn
from experiment_utils import (
    add_output_arguments,
    load_lab_data,
    prepare_output_dir,
    save_run_artifacts,
    set_all_seeds,
)
from training import train_dino


TUNING_CONFIGS = (
    {"std_mode": "scalar", "tau": 0.08, "K": 1024},
    {"std_mode": "scalar", "tau": 0.10, "K": 1024},
    {"std_mode": "scalar", "tau": 0.12, "K": 1024},
    {"std_mode": "column", "tau": 0.08, "K": 1024},
    {"std_mode": "column", "tau": 0.10, "K": 1024},
    {"std_mode": "scalar", "tau": 0.10, "K": 512},
)


def standardized_targets(t1, t2, *, tau=0.1, std_mode="scalar", eps=1e-5):
    """Convert two teacher-logit batches into standardized softmax targets.

    Concatenate inputs of shape ``(batch, K)`` into ``(2 * batch, K)`` and
    subtract the current per-column mean. With ``std_mode="scalar"``, divide
    by one standard deviation computed over all concatenated logits; with
    ``"column"``, use one standard deviation per column. Both modes use
    ``correction=0`` and compute ``softmax((logits - mean) / (std + eps) / tau)``.

    Return ``(p1, p2, mean, std)``. Each probability tensor has shape
    ``(batch, K)``, the mean has shape ``(K,)``, and std is a scalar tensor or
    a tensor of shape ``(K,)`` depending on the mode. Statistics are computed
    from raw logits before centering. This function does not detach its outputs.
    """
    if t1.ndim != 2 or t1.shape != t2.shape:
        raise ValueError("Teacher logits must have matching (batch, K) shapes.")
    if tau <= 0 or eps <= 0:
        raise ValueError("tau and eps must be positive.")
    if std_mode not in {"scalar", "column"}:
        raise ValueError(f"Unknown std_mode: {std_mode}")

    logits = torch.cat((t1, t2), dim=0)
    mean = logits.mean(dim=0, keepdim=True)
    if std_mode == "column":
        std = logits.std(dim=0, keepdim=True, correction=0)
    else:
        std = logits.std(correction=0)
    probabilities = F.softmax((logits - mean) / (std + eps) / tau, dim=-1)

    batch_size = t1.shape[0]
    return (
        probabilities[:batch_size],
        probabilities[batch_size:],
        mean.squeeze(0),
        std.squeeze(0) if std_mode == "column" else std,
    )


def train_dino_standardize(
    train_images,
    train_labels,
    *,
    epochs=20,
    seed=0,
    K=1024,
    batch_size=512,
    lr=1e-3,
    tau=0.1,
    std_mode="scalar",
    ema=0.996,
):
    """Train DINO with current-batch standardized teacher targets.

    Reuse ``train_dino`` with its running center disabled and a shared ``tau``
    for teacher and student softmaxes. The target transform standardizes teacher
    logits across both views; student logits are only temperature-scaled.

    Return the usual DINO result plus detached ``target_mean`` and ``target_std``
    tensors averaged over training batches, and ``last_target_mean`` and
    ``last_target_std`` from the final batch. These describe raw teacher logits;
    the averaged statistics are for diagnostics and do not define later targets.
    The mean has shape ``(K,)``; std is scalar or ``(K,)`` according to std_mode.
    """
    if not 1 <= epochs <= 20:
        raise ValueError("Each run must use between 1 and 20 epochs.")
    if batch_size < 1 or lr <= 0 or not 0 <= ema < 1:
        raise ValueError("batch_size, lr, or ema is outside its valid range.")
    if len(train_images) // batch_size == 0:
        raise ValueError("batch_size must not exceed the number of training images.")
    if K < 1:
        raise ValueError("K must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if std_mode not in {"scalar", "column"}:
        raise ValueError(f"Unknown std_mode: {std_mode}")

    set_all_seeds(seed)
    statistics = {}
    stat_batches = 0

    def make_targets(t1, t2):
        """Build both target distributions and accumulate detached batch moments."""
        nonlocal stat_batches
        target1, target2, mean, std = standardized_targets(
            t1, t2, tau=tau, std_mode=std_mode
        )
        for name, value in (("mean", mean), ("std", std)):
            value = value.detach()
            key = f"target_{name}"
            if stat_batches == 0:
                statistics[key] = value.clone()
            else:
                statistics[key].add_(value)
            statistics[f"last_target_{name}"] = value
        stat_batches += 1
        return target1, target2

    result = train_dino(
        train_images,
        train_labels,
        center=False,
        sharpen=False,
        epochs=epochs,
        seed=seed,
        K=K,
        batch_size=batch_size,
        lr=lr,
        tau_s=tau,
        ema=ema,
        target_transform=make_targets,
    )
    for key in ("target_mean", "target_std"):
        statistics[key] = (statistics[key] / stat_batches).detach()
    result.update(statistics)
    return result


def _configuration_name(configuration):
    return (
        f"standardize_{configuration['std_mode']}_"
        f"tau{configuration['tau']}_K{configuration['K']}"
    )


def run_experiments(output_dir):
    train_images, train_labels, test_images, test_labels = load_lab_data()
    best_config = None
    best_result = None
    best_val = -1.0

    print("Starting Person 2 tuning grid (six configurations, seed 0).")
    for configuration in TUNING_CONFIGS:
        tag = _configuration_name(configuration)
        print(f"\n===== Evaluating {tag} =====", flush=True)
        result = train_dino_standardize(
            train_images,
            train_labels,
            epochs=20,
            seed=0,
            **configuration,
        )
        val_accuracy = val_knn(
            result["student"].backbone, train_images, train_labels
        )
        save_run_artifacts(
            output_dir,
            artifact_name=f"{tag}_seed0_tuning",
            method="Standardize",
            seed=0,
            result=result,
            configuration=configuration,
            val_acc=val_accuracy,
            test_acc=None,
            note="tuning",
        )
        print(f"{tag}: val_knn={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_val:
            best_config = dict(configuration)
            best_result = result
            best_val = val_accuracy

    if best_config is None or best_result is None:
        raise RuntimeError("The standardization tuning grid produced no result.")
    print(f"\nSelected Person 2 configuration: {best_config} (val={best_val:.4f})")

    # Test the selected seed-0 model only after tuning has finished.
    best_test = knn_eval(
        best_result["student"].backbone,
        train_images,
        train_labels,
        test_images,
        test_labels,
    )
    save_run_artifacts(
        output_dir,
        artifact_name="standardize_best_seed0",
        method="Standardize",
        seed=0,
        result=best_result,
        configuration=best_config,
        val_acc=best_val,
        test_acc=best_test,
        note="final_eval",
    )
    print(f"seed 0: val_knn={best_val:.4f}, knn_eval={best_test:.4f}")

    for seed in (1, 2):
        print(f"\n===== Final Person 2 run, seed {seed} =====", flush=True)
        result = train_dino_standardize(
            train_images,
            train_labels,
            epochs=20,
            seed=seed,
            **best_config,
        )
        val_accuracy = val_knn(
            result["student"].backbone, train_images, train_labels
        )
        test_accuracy = knn_eval(
            result["student"].backbone,
            train_images,
            train_labels,
            test_images,
            test_labels,
        )
        save_run_artifacts(
            output_dir,
            artifact_name=f"standardize_best_seed{seed}",
            method="Standardize",
            seed=seed,
            result=result,
            configuration=best_config,
            val_acc=val_accuracy,
            test_acc=test_accuracy,
            note="final_eval",
        )
        print(
            f"seed {seed}: val_knn={val_accuracy:.4f}, "
            f"knn_eval={test_accuracy:.4f}",
            flush=True,
        )
    return best_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    output_dir = prepare_output_dir(
        args.output_dir, mount_drive=not args.no_drive_mount
    )
    run_experiments(output_dir)


if __name__ == "__main__":
    main()

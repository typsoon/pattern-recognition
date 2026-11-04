"""Person 2: DINO with target-side batch moment standardization."""

import argparse
import copy
import time

import torch
from torch import nn
from torch.nn import functional as F

from common import device
from evaluation import knn_eval, val_knn
from experiment_utils import (
    add_output_arguments,
    load_lab_data,
    prepare_output_dir,
    save_run_artifacts,
    set_all_seeds,
)
from network import DINONetwork
from training import augment, collapse_curves


TUNING_CONFIGS = (
    {"std_mode": "scalar", "tau": 0.08, "K": 1024},
    {"std_mode": "scalar", "tau": 0.10, "K": 1024},
    {"std_mode": "scalar", "tau": 0.12, "K": 1024},
    {"std_mode": "column", "tau": 0.08, "K": 1024},
    {"std_mode": "column", "tau": 0.10, "K": 1024},
    {"std_mode": "scalar", "tau": 0.10, "K": 512},
)


def standardized_targets(t1, t2, *, tau=0.1, std_mode="scalar", eps=1e-5):
    """Standardize both views with shared batch statistics, then split targets."""
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
    student = DINONetwork(K).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 200)
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    running_mean = None
    running_std = None
    last_mean = None
    last_std = None
    stat_batches = 0
    step = 0
    start = time.time()
    center = torch.zeros(K, device=device)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_images), device=device)
        batch_count = len(order) // batch_size
        for batch_index in range(batch_count):
            batch_indices = order[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            images = train_images[batch_indices]
            view1, view2 = augment(images), augment(images)

            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    teacher1, teacher2 = teacher(view1), teacher(view2)
                student1, student2 = student(view1), student(view2)

            teacher1, teacher2 = teacher1.float(), teacher2.float()
            student1, student2 = student1.float(), student2.float()
            target1, target2, batch_mean, batch_std = standardized_targets(
                teacher1, teacher2, tau=tau, std_mode=std_mode
            )
            log_student1 = F.log_softmax(student1 / tau, dim=-1)
            log_student2 = F.log_softmax(student2 / tau, dim=-1)
            loss = 0.5 * (
                -(target1 * log_student2).sum(dim=-1).mean()
                -(target2 * log_student1).sum(dim=-1).mean()
            )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            warmup.step()

            step += 1
            if step % 20 == 0 or batch_index == batch_count - 1:
                history.append(
                    {
                        "step": step,
                        "loss": loss.item(),
                        **collapse_curves(target1, log_student2),
                    }
                )

            with torch.no_grad():
                for teacher_parameter, student_parameter in zip(
                    teacher.parameters(), student.parameters()
                ):
                    teacher_parameter.mul_(ema).add_(
                        student_parameter, alpha=1.0 - ema
                    )
                if running_mean is None:
                    running_mean = batch_mean.detach().clone()
                    running_std = batch_std.detach().clone()
                else:
                    running_mean.add_(batch_mean.detach())
                    running_std.add_(batch_std.detach())
                last_mean = batch_mean.detach()
                last_std = batch_std.detach()
                stat_batches += 1

        last = history[-1]
        validation = ""
        if epoch % 5 == 0:
            accuracy = val_knn(student.backbone, train_images, train_labels)
            validation = f", val_knn {accuracy:.4f}"
        print(
            f"epoch {epoch:2d} ({time.time() - start:4.0f}s): "
            f"entropy {last['entropy']:.2f}, "
            f"batch entropy {last['batch_entropy']:.2f}, "
            f"KL {last['kl']:.3f}{validation}",
            flush=True,
        )

    return {
        "student": student,
        "teacher": teacher,
        "center": center,
        "tau_t": tau,
        "history": history,
        "target_mean": (running_mean / stat_batches).detach(),
        "target_std": (running_std / stat_batches).detach(),
        "last_target_mean": last_mean,
        "last_target_std": last_std,
    }


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

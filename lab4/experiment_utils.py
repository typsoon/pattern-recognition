"""Shared data, reproducibility, and artifact helpers for Task 4 experiments."""

import argparse
import csv
import errno
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common import load


DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/dino_task4")
RESULT_FIELDS = ("method", "seed", "val_knn", "knn_eval", "timestamp", "note")
LAB_DIR = Path(__file__).resolve().parent


def add_output_arguments(parser):
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory (defaults to the shared Google Drive folder).",
    )
    parser.add_argument(
        "--no-drive-mount",
        action="store_true",
        help="Skip Colab Drive mounting; supply --output-dir for a local path.",
    )


def prepare_output_dir(output_dir, *, mount_drive=True):
    if mount_drive:
        try:
            from google.colab import drive
        except ImportError as error:
            raise RuntimeError(
                "Google Drive mounting requires Colab. Use --no-drive-mount "
                "and provide --output-dir when running elsewhere."
            ) from error
        drive.mount("/content/drive", force_remount=False)
    elif Path(output_dir).expanduser() == DEFAULT_OUTPUT_DIR:
        raise ValueError(
            "Specify --output-dir when using --no-drive-mount."
        )

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    return output_dir


def load_lab_data():
    """Load both CIFAR-10 splits from lab4, independent of the launch directory."""
    original_directory = Path.cwd()
    try:
        os.chdir(LAB_DIR)
        train_images, train_labels = load("train")
        test_images, test_labels = load("test")
    finally:
        os.chdir(original_directory)
    return train_images, train_labels, test_images, test_labels


def set_all_seeds(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def _exclusive_file_lock(lock_path):
    """Serialize CSV appends across independent local experiment processes."""
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {
                        errno.EACCES,
                        errno.EDEADLK,
                        errno.EAGAIN,
                    }:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_result(output_dir, *, method, seed, val_acc, test_acc, note):
    if val_acc is not None and not 0 <= float(val_acc) <= 1:
        raise ValueError("val_knn must be an accuracy between 0 and 1.")
    if test_acc is not None and not 0 <= float(test_acc) <= 1:
        raise ValueError("knn_eval must be an accuracy between 0 and 1.")

    results_path = Path(output_dir) / "results.csv"
    lock_path = results_path.with_suffix(".csv.lock")
    row = {
        "method": method,
        "seed": int(seed),
        "val_knn": "" if val_acc is None else f"{float(val_acc):.8f}",
        "knn_eval": "" if test_acc is None else f"{float(test_acc):.8f}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }

    with _exclusive_file_lock(lock_path):
        needs_header = not results_path.exists() or results_path.stat().st_size == 0
        with results_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    return results_path


def _cpu_state_dict(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _atomic_torch_save(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(path, value):
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_run_artifacts(
    output_dir,
    *,
    artifact_name,
    method,
    seed,
    result,
    configuration,
    val_acc,
    test_acc,
    note,
):
    """Save portable weights/history/metadata and append one shared CSV row."""
    if Path(artifact_name).name != artifact_name:
        raise ValueError("artifact_name must be a plain filename stem.")
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"{artifact_name}.pt"
    history_path = checkpoints_dir / f"{artifact_name}_history.json"
    summary_path = checkpoints_dir / f"{artifact_name}_summary.json"
    student = result["student"]
    teacher = result["teacher"]
    checkpoint = {
        "method": method,
        "seed": int(seed),
        "configuration": configuration,
        "student_state": _cpu_state_dict(student),
        "teacher_state": _cpu_state_dict(teacher),
        "student_backbone": _cpu_state_dict(student.backbone),
        "teacher_backbone": _cpu_state_dict(teacher.backbone),
        "center": (
            result["center"].detach().cpu()
            if isinstance(result.get("center"), torch.Tensor)
            else result.get("center")
        ),
        "tau_t": result.get("tau_t", result.get("tau")),
    }
    for key in ("target_mean", "target_std", "last_target_mean", "last_target_std"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        if value is not None:
            checkpoint[key] = value

    _atomic_torch_save(checkpoint_path, checkpoint)
    _atomic_json_save(history_path, result["history"])
    _atomic_json_save(
        summary_path,
        {
            "method": method,
            "seed": int(seed),
            "configuration": configuration,
            "val_knn": val_acc,
            "knn_eval": test_acc,
            "note": note,
            "checkpoint": checkpoint_path.name,
            "history": history_path.name,
        },
    )
    append_result(
        output_dir,
        method=method,
        seed=seed,
        val_acc=val_acc,
        test_acc=test_acc,
        note=note,
    )
    return checkpoint_path, history_path, summary_path

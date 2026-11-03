import logging
from pathlib import Path

import torch
from common import device

CHECKPOINT_DIR = Path.cwd() / "checkpoints"  # Keep saved runs beside the experiment script.


def cached_train_dino(train_fn, make_network, name=None, **kwargs):
    # Same as train_fn(**kwargs), but loads checkpoints/<name>.pt instead of training when it exists.
    # train_fn     - a DINO loop returning {"student", "teacher", "center", "tau_t", "history"}
    # make_network - builds an empty network from K, e.g. DINONetwork, to load the saved weights into
    # name defaults to one built from the kwargs, so every configuration gets its own file.
    # Sort options so their argument order does not change the checkpoint filename.
    # An explicit name overrides this automatic naming scheme.
    name = name or "dino" + "".join(f"_{k}={v}" for k, v in sorted(kwargs.items()))
    path = CHECKPOINT_DIR / f"{name}.pt"
    if path.exists():  # Reuse a completed run instead of spending time training again.
        logging.info("Loading %s", path)
        saved = torch.load(path, map_location=device)  # Move saved tensors to this device.
        # Recreate architectures with the saved number of output codes before loading weights.
        student = make_network(saved["K"]).to(device)
        teacher = make_network(saved["K"]).to(device)
        student.load_state_dict(saved["student"])  # Restore every learned student parameter.
        teacher.load_state_dict(saved["teacher"])  # Restore the teacher's EMA parameters.
        teacher.requires_grad_(False)  # The teacher is updated without backpropagation.
        # Match train_dino's result structure so plotting works for cached and fresh runs.
        return {
            "student": student,
            "teacher": teacher,
            "center": saved["center"],
            "tau_t": saved["tau_t"],
            "history": saved["history"],
        }
    run = train_fn(**kwargs)  # Train only when the requested checkpoint is absent.
    CHECKPOINT_DIR.mkdir(exist_ok=True)  # Create the output directory on the first run.
    # Save weights and diagnostics; optimizer state is unnecessary for completed runs.
    torch.save(
        {
            "K": run["center"].shape[0],  # Needed to reconstruct the projection heads.
            "student": run["student"].state_dict(),  # Store tensors rather than model objects.
            "teacher": run["teacher"].state_dict(),
            "center": run["center"],  # Reproduce centering with the final running mean.
            "tau_t": run["tau_t"],  # Preserve the effective temperature, including ablations.
            "history": run["history"],  # Keep all logged points for the training curves.
        },
        path,
    )
    logging.info("Saved %s", path)
    return run

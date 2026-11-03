import logging
from pathlib import Path

import torch
from common import device

CHECKPOINT_DIR = Path.cwd() / "checkpoints"


def cached_train_dino(train_fn, make_network, name=None, **kwargs):
    # Same as train_fn(**kwargs), but loads checkpoints/<name>.pt instead of training when it exists.
    # train_fn     - a DINO loop returning {"student", "teacher", "center", "tau_t", "history"}
    # make_network - builds an empty network from K, e.g. DINONetwork, to load the saved weights into
    # name defaults to one built from the kwargs, so every configuration gets its own file.
    name = name or "dino" + "".join(f"_{k}={v}" for k, v in sorted(kwargs.items()))
    path = CHECKPOINT_DIR / f"{name}.pt"
    if path.exists():
        logging.info("Loading %s", path)
        saved = torch.load(path, map_location=device)
        student = make_network(saved["K"]).to(device)
        teacher = make_network(saved["K"]).to(device)
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        teacher.requires_grad_(False)
        return {
            "student": student,
            "teacher": teacher,
            "center": saved["center"],
            "tau_t": saved["tau_t"],
            "history": saved["history"],
        }
    run = train_fn(**kwargs)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "K": run["center"].shape[0],
            "student": run["student"].state_dict(),
            "teacher": run["teacher"].state_dict(),
            "center": run["center"],
            "tau_t": run["tau_t"],
            "history": run["history"],
        },
        path,
    )
    logging.info("Saved %s", path)
    return run

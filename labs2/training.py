"""Everything the experiments (B, C, D, E) share: data loaders, the loss, the training loop
and checkpointing.

Checkpoints and logs go to Google Drive when running on Colab (so nothing is lost if the
kernel restarts), otherwise to a local `checkpoints/` folder.
"""
import json
import os
import time

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from common import MAX_IMGS, cfg, device, to_imagenet

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SAVE_DIR = "/content/drive/MyDrive/deit_seminar_task2"
except ImportError:
    SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"All checkpoints and logs will be saved to: {SAVE_DIR}")


# ── Data ────────────────────────────────────────────────────

def make_loader(resolution, batch_size=64):
    """Imagewoof train loader with the model's own preprocessing and augmentation."""
    tf = timm.data.create_transform(**dict(cfg, input_size=(3, resolution, resolution), is_training=True))
    ds = ImageFolder("imagewoof2/train", transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# ── Models ──────────────────────────────────────────────────

def train_only(model, *parts):
    """Freeze the whole model, then unfreeze `parts` (modules or parameters)."""
    model.requires_grad_(False)
    for part in parts:
        part.requires_grad_(True)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.2f}%)")
    return trainable


# ── Loss ────────────────────────────────────────────────────

def imagenet_ce_loss(model, x, y):
    """CE over the 10 Imagewoof logits of the 1000-class model, the same slice evaluate() scores."""
    return F.cross_entropy(model(x)[:, to_imagenet], y)


# ── Training loop ───────────────────────────────────────────

def train(model, loader, loss_fn, optimizer, epochs, scheduler=None, eval_fn=None, label=""):
    """Train with mixed precision and grad clipping; returns the per-epoch history.

    loss_fn(model, x, y) -> scalar loss. eval_fn(model) -> accuracy runs after every epoch.
    """
    images = epochs * len(loader.dataset)
    assert images <= MAX_IMGS, f"Budget exceeded: {images:,} > {MAX_IMGS:,} images"

    scaler = torch.amp.GradScaler("cuda")
    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model, x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(x)
            n += len(x)
        if scheduler is not None:
            scheduler.step()

        entry = {"epoch": ep, "loss": total_loss / n, "images": n}
        msg = f"[{label}] ep {ep:2d}/{epochs}  loss={entry['loss']:.4f}"
        if eval_fn is not None:
            entry["val_acc"] = eval_fn(model)
            msg += f"  val={entry['val_acc']:.2%}"
        history.append(entry)
        print(f"{msg}  ({(time.time() - t0) / 60:.1f} min)")

    return history


# ── Checkpoints ─────────────────────────────────────────────

def load_checkpoint(model, model_file, log_file):
    """Load saved weights into `model` and return the saved history, or None if there is none."""
    model_path = os.path.join(SAVE_DIR, model_file)
    log_path = os.path.join(SAVE_DIR, log_file)
    if not (os.path.exists(model_path) and os.path.exists(log_path)):
        return None
    print(f"Found saved checkpoint {model_path}. Loading parameters and logs...")
    model.load_state_dict(torch.load(model_path))
    with open(log_path) as f:
        history = json.load(f)
    print("Loaded validation history:", history)
    return history


def save_checkpoint(model, history, model_file, log_file):
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_file))
    with open(os.path.join(SAVE_DIR, log_file), "w") as f:
        json.dump(history, f)

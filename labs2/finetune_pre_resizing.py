# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Fine-tuning control @224 (before any resizing)
#
# **Author: Piotr**
#
# The same model and the same 11-epoch budget as the @384 runs, but at the original resolution,
# so "fine-tuned @384 vs fine-tuned @224" isolates the effect of the extra pixels.
# `Deit_starter.py` calls `run_control`; running this file directly does the same.

# %%
import copy

import torch

from common import evaluate, save_result
from training import imagenet_ce_loss, make_loader, train

# Fine-tuning control: 11 epochs = 99,275 images, fitting the 100k budget (train() asserts it)
EPOCHS = 11


# %%
def run_control(base_model, resolution=224):
    """Fine-tune a copy of `base_model` at `resolution`; `base_model` itself is left untouched."""
    copied_model = copy.deepcopy(base_model)

    optimizer = torch.optim.AdamW(copied_model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Starting {resolution}x{resolution} Fine-Tuning Control Run...")
    train(copied_model, make_loader(resolution), imagenet_ce_loss, optimizer, EPOCHS, scheduler, label="control")

    acc = evaluate(copied_model, resolution)
    print(f"DeiT-S fine-tuned @{resolution}: top-1 {acc:.2%}")
    save_result(f"finetune_control_{resolution}", acc)
    return copied_model, acc


# %%
if __name__ == "__main__":
    from common import load_pretrained
    run_control(load_pretrained())

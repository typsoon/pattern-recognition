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
# # Solution C — Full fine-tuning @384
#
# **Author: Vitaliy**
#
# We take a clean copy of the baseline model, scale the position embeddings to $24 \times 24$ ($384$ px),
# configure differential learning rates, and train for 11 epochs (budget limit of 99,275 training images total).
# If a saved checkpoint exists, we load it immediately to avoid unnecessary retraining.
# `Deit_starter.py` calls `run_c`; running this file directly does the same.
#
# Original run (Colab T4), fixed `evaluate()` @384: **97.58%** (baseline @224: 97.20%).

# %%
import copy

import torch.optim as optim

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import imagenet_ce_loss, load_checkpoint, make_loader, save_checkpoint, train


# %%
def run_c(base_model, resolution=384):
    # Instantiate a clean copy of the baseline model, everything trainable
    model_384_full = resize_pos_embed_(copy.deepcopy(base_model), resolution).requires_grad_(True)

    history_c = load_checkpoint(model_384_full, "model_384_full.pth", "history_full.json")
    if history_c is None:
        params = dict(model_384_full.named_parameters())
        optimizer_c = optim.AdamW([
            {'params': [p for n, p in params.items() if 'head' not in n and 'pos_embed' not in n], 'lr': 1e-5},
            {'params': [p for n, p in params.items() if 'pos_embed' in n], 'lr': 1e-4},
            {'params': [p for n, p in params.items() if 'head' in n], 'lr': 1e-4},
        ], weight_decay=0.05)
        scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=MAX_EPOCHS)

        history_c = train(model_384_full, make_loader(resolution), imagenet_ce_loss, optimizer_c, MAX_EPOCHS,
                          scheduler_c, eval_fn=lambda m: evaluate(m, resolution), label="C")
        save_checkpoint(model_384_full, history_c, "model_384_full.pth", "history_full.json")

    acc_c = evaluate(model_384_full, resolution)
    print(f"Solution C (Full Fine-Tuning @{resolution}): {acc_c:.2%}")
    save_result(f"c_full_finetune_{resolution}", acc_c)
    return model_384_full, acc_c


# %%
if __name__ == "__main__":
    from common import load_pretrained
    run_c(load_pretrained())

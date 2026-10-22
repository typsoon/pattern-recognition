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
# # Solution D — Layer freezing @384
#
# **Author: Gabriela**
#
# Starts from a copy of the given 1000-class model and is scored with the fixed `evaluate()`.
# `Deit_starter.py` calls `run_d`; running this file directly does the same.

# %%
import copy

import torch

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import count_params, imagenet_ce_loss, make_loader, train, train_only


# %%
# ════════════════════════════════════════════════════════════
#  SOLUTION D  —  Layer Freezing @ 384
#
#  Architecture:
#    patch_embed + blocks 0–7  → FROZEN   (8 / 12 blocks)
#    pos_embed + blocks 8–11   → trainable
#    norm + head + head_dist   → trainable
#
#  WHY freeze early blocks?
#    Blocks 0–7 learn low-level features (edges, textures, local
#    patches) that are largely resolution-agnostic and already
#    well-trained on ImageNet.  Keeping them frozen:
#      (a) cuts trainable params → far less overfitting risk
#          on our 9 025-image dataset;
#      (b) preserves the strong ImageNet representations.
#    Blocks 8–11 learn high-level, global semantics that benefit
#    from 384×384's extra context — they need to adapt.
# ════════════════════════════════════════════════════════════
def run_d(base_model, resolution=384, freeze_blocks=8):
    model_d = resize_pos_embed_(copy.deepcopy(base_model), resolution)

    # freeze blocks 0 … freeze_blocks-1  (default: first 8 of 12)
    train_only(model_d, model_d.pos_embed, model_d.blocks[freeze_blocks:],
               model_d.norm, model_d.head, model_d.head_dist)
    count_params(model_d)

    # ── Differential learning rates ──────────────────────────────
    # pos_embed + backbone tail → moderate (pre-trained weights, gentle update)
    # heads                     → higher
    opt_d = torch.optim.AdamW(
        [
            {"params": [model_d.pos_embed], "lr": 5e-5},
            {"params": model_d.norm.parameters(), "lr": 5e-5},
            {"params": model_d.blocks[freeze_blocks:].parameters(), "lr": 5e-5},
            {"params": model_d.head.parameters(), "lr": 5e-4},
            {"params": model_d.head_dist.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.05,
    )
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, MAX_EPOCHS, eta_min=1e-6)

    train(model_d, make_loader(resolution, batch_size=32), imagenet_ce_loss, opt_d, MAX_EPOCHS, sch_d,
          eval_fn=lambda m: evaluate(m, resolution), label="D")
    acc_d = evaluate(model_d, resolution)
    print(f"D layer freezing @{resolution}: top-1 {acc_d:.2%}")
    save_result(f"d_layer_freezing_{resolution}", acc_d)
    return model_d, acc_d


# %%
if __name__ == "__main__":
    from common import load_pretrained
    model = load_pretrained()
    run_d(model)

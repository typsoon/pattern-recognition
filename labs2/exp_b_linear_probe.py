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
# # Solution B — Linear probe @384
#
# **Author: Gabriela**
#
# - **B1:** only `head` + `head_dist` are trained; backbone and the resized pos_embed stay frozen.
# - **B2:** the same, plus `pos_embed` is trained with a 10× smaller LR.
#
# Both start from a copy of the given 1000-class model and are scored with the fixed `evaluate()`.
# `Deit_starter.py` calls `run_b1` / `run_b2`; running this file directly does the same.

# %%
import copy

import torch

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import count_params, imagenet_ce_loss, make_loader, train, train_only


# %%
# ════════════════════════════════════════════════════════════
#  SOLUTION B1  —  Linear Probe @ 384
#  Trainable: head + head_dist
#  Backbone + pos_embed: fully frozen
# ════════════════════════════════════════════════════════════
def run_b1(base_model, resolution=384):
    model_b1 = resize_pos_embed_(copy.deepcopy(base_model), resolution)   # pos_embed interpolated from 14×14 to 24×24
    train_only(model_b1, model_b1.head, model_b1.head_dist)
    count_params(model_b1)

    # High LR is safe: only 2 linear layers, no backbone risk
    opt_b1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_b1.parameters()),
        lr=1e-3, weight_decay=0.05
    )
    sch_b1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b1, MAX_EPOCHS, eta_min=1e-5)

    train(model_b1, make_loader(resolution, batch_size=32), imagenet_ce_loss, opt_b1, MAX_EPOCHS, sch_b1,
          eval_fn=lambda m: evaluate(m, resolution), label="B1")
    acc_b1 = evaluate(model_b1, resolution)
    print(f"B1 linear probe @{resolution}: top-1 {acc_b1:.2%}")
    save_result(f"b1_linear_probe_{resolution}", acc_b1)
    return model_b1, acc_b1


# %%
# ════════════════════════════════════════════════════════════
#  SOLUTION B2  —  Linear Probe + pos_embed @ 384
#  Trainable: head + head_dist + pos_embed
#
#  WHY unfreeze pos_embed here?
#  Bicubic interpolation gives geometrically consistent values,
#  but the frozen attention layers were never trained with 24×24
#  positions — the spatial "grammar" is mismatched.
#  Letting pos_embed fine-tune with a small LR corrects that
#  mismatch without touching the backbone weights, which would
#  risk overfitting on our tiny 9 k dataset.
# ════════════════════════════════════════════════════════════
def run_b2(base_model, resolution=384):
    model_b2 = resize_pos_embed_(copy.deepcopy(base_model), resolution)
    train_only(model_b2, model_b2.head, model_b2.head_dist,
               model_b2.pos_embed)     # ← the key difference vs B1
    count_params(model_b2)

    # Separate LR groups: lower rate for pos_embed to stay conservative
    opt_b2 = torch.optim.AdamW([
        {"params": model_b2.head.parameters(),      "lr": 1e-3},
        {"params": model_b2.head_dist.parameters(), "lr": 1e-3},
        {"params": [model_b2.pos_embed],            "lr": 1e-4},  # ← 10× smaller
    ], weight_decay=0.05)
    sch_b2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b2, MAX_EPOCHS, eta_min=1e-6)

    train(model_b2, make_loader(resolution, batch_size=32), imagenet_ce_loss, opt_b2, MAX_EPOCHS, sch_b2,
          eval_fn=lambda m: evaluate(m, resolution), label="B2")
    acc_b2 = evaluate(model_b2, resolution)
    print(f"B2 linear probe + pos_embed @{resolution}: top-1 {acc_b2:.2%}")
    save_result(f"b2_linear_probe_pos_embed_{resolution}", acc_b2)
    return model_b2, acc_b2


# %%
if __name__ == "__main__":
    from common import load_pretrained
    model = load_pretrained()
    run_b1(model)
    run_b2(model)

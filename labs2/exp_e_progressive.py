# # Solution E — Progressive resolution curriculum
#
# **Author: Vitaliy**
#
# Instead of jumping straight to 384, we transition through `224 -> 256 -> 320 -> 384`.
# To avoid positional embedding degradation (blur) from sequential interpolation, we always keep
# a pristine 224 model copy and interpolate directly from it to the targeted grid sizes.
# `Deit_starter.py` calls `run_e`; running this file directly does the same.
#
# Original run (Colab T4), fixed `evaluate()` @384: **97.38%** (baseline @224: 97.20%).

import copy

import torch.optim as optim

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import imagenet_ce_loss, load_checkpoint, make_loader, save_checkpoint, train

# Schedule designed to respect the budget (<100,000 images total)
SCHEDULE = [
    (224, 4, 5e-5),  # 4 epochs
    (256, 4, 2e-5),  # 4 epochs
    (320, 1, 1e-5),  # 1 epoch
    (384, 2, 5e-6),  # 2 epochs
]  # Total 11 epochs = ~99,275 images


def run_e(base_model, schedule=SCHEDULE):
    """Train a copy of `base_model` through `schedule` = [(resolution, epochs, lr), ...]."""
    assert sum(epochs for _, epochs, _ in schedule) <= MAX_EPOCHS
    final_res = schedule[-1][0]

    model_prog = resize_pos_embed_(copy.deepcopy(base_model), final_res)
    history_e = load_checkpoint(model_prog, "model_prog.pth", "history_prog.json")

    if history_e is None:
        model_prog = copy.deepcopy(base_model)
        history_e = []
        for resolution, epochs, lr in schedule:
            print(f"\nTransitioning to Resolution: {resolution} (Grid: {resolution // 16}x{resolution // 16}) with LR: {lr}")

            # Interpolate cleanly from the original model to prevent repeated interpolation blur,
            # then copy the current updated weights over (excluding pos_embed)
            next_model = resize_pos_embed_(copy.deepcopy(base_model), resolution)
            state = {k: v for k, v in model_prog.state_dict().items() if k != 'pos_embed'}
            next_model.load_state_dict(state, strict=False)
            model_prog = next_model.requires_grad_(True)

            optimizer_e = optim.AdamW(model_prog.parameters(), lr=lr, weight_decay=0.05)
            stage = train(model_prog, make_loader(resolution), imagenet_ce_loss, optimizer_e, epochs,
                          eval_fn=lambda m: evaluate(m, resolution), label=f"E@{resolution}")
            history_e += [dict(h, res=resolution) for h in stage]

        save_checkpoint(model_prog, history_e, "model_prog.pth", "history_prog.json")

    acc_e = evaluate(model_prog, final_res)
    print(f"Solution E (Progressive Curr @{final_res}): {acc_e:.2%}")
    save_result(f"e_progressive_{final_res}", acc_e)
    return model_prog, acc_e


if __name__ == "__main__":
    from common import load_pretrained
    run_e(load_pretrained())

# # Solution A — 0-shot @384
#
# **Author: Piotr**
#
# Resize the position embeddings 14×14 → 24×24 and evaluate without any training.
# `Deit_starter.py` calls `run_zero_shot`; running this file directly does the same for all modes.

import copy
from common import evaluate, resize_pos_embed_, save_result

MODES = ['nearest', 'bilinear', 'bicubic']


def run_zero_shot(base_model, resolution=384, mode='bicubic'):
    """Evaluate a copy of `base_model` with pos_embed resized for `resolution`, no training."""
    print(f"Chosen interpolation mode: {mode}")
    print(base_model.pos_embed.shape, "Before resizing")   # torch.Size([1, 198, 384])

    # resize a copy, so `base_model` stays the pretrained 224 model
    model_0shot = resize_pos_embed_(copy.deepcopy(base_model), resolution, mode)
    print(model_0shot.pos_embed.shape, "After resizing")   # torch.Size([1, 578, 384]) at 384

    acc = evaluate(model_0shot, resolution)
    print(f"DeiT-S 0-shot ({mode}) @{resolution}: top-1 {acc:.2%}")
    save_result(f"zero_shot_{mode}_{resolution}", acc)
    return model_0shot, acc


if __name__ == "__main__":
    from common import load_pretrained
    model = load_pretrained()
    for mode in MODES:
        run_zero_shot(model, 384, mode)

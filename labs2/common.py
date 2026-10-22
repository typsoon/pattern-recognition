"""Shared setup for task 2: data, the pretrained DeiT-S, `to_imagenet` and the fixed `evaluate()`.

This is the course starter's setup code (unchanged), moved here so `Deit_starter.py`, the experiment
files and `training.py` all share one copy, plus our position-embedding resizing. Importing it downloads Imagewoof (if missing); it doesn't
load the model, call `load_pretrained()` for that.
"""

import json
import os
import tarfile
import urllib.request
from math import isqrt

import einops
import torch
import torch.nn.functional as F
import timm
from torch import nn
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", (
    "Switch to a GPU runtime: Runtime -> Change runtime type -> T4 GPU"
)

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"
# we use the full-size image sets on purpose, not the smaller "-320" versions
# can you tell why? (in the deck)
if not os.path.exists("imagewoof2/val"):
    urllib.request.urlretrieve(URL, "imagewoof2.tgz")
    with tarfile.open("imagewoof2.tgz") as t:
        t.extractall(filter="data")
for split in ["train", "val"]:
    print(split, sum(len(f) for _, _, f in os.walk(f"imagewoof2/{split}")), "photos")

N_TRAIN = 9_025  # imagewoof train set size
MAX_IMGS = 100_000  # fine-tuning budget per model
MAX_EPOCHS = MAX_IMGS // N_TRAIN  # 11

MODEL_NAME = "deit_small_distilled_patch16_224"


def load_pretrained():
    """The given starting model: DeiT-S distilled, trained on ImageNet at 224."""
    return timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()


# the preprocessing the model was trained with, read from timm's config without loading the weights
cfg = timm.data.resolve_data_config(
    pretrained_cfg=timm.models.get_pretrained_cfg(MODEL_NAME).to_dict()
)

# ImageFolder labels the 10 folders 0..9, but the model outputs 1000 ImageNet classes:
# to_imagenet[label] translates a folder label into the ImageNet class index
wnids = list(timm.data.ImageNetInfo().label_names())
breeds = datasets.ImageFolder("imagewoof2/val").classes
to_imagenet = torch.tensor([wnids.index(w) for w in breeds], device=device)
print(cfg)
print("folder -> ImageNet index:", dict(zip(breeds, to_imagenet.tolist())))


@torch.no_grad()
def evaluate(model, resolution):
    """The fixed evaluation for task 2. Do not change it; report exactly what it prints.

    model      - any model that takes (batch, 3, resolution, resolution) images and returns 1000 logits
    resolution - the input size: 224 for the original model, e.g. 384 for yours
    returns: accuracy - top-1 over the 10 breeds, on the 3,929 Imagewoof validation photos
    """
    model.eval()
    tf = timm.data.create_transform(**dict(cfg, input_size=(3, resolution, resolution)))
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("imagewoof2/val", transform=tf),
        batch_size=64,
        num_workers=2,
    )
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        correct += (
            (logits[:, to_imagenet].argmax(-1) == y).sum().item()
        )  # best of the 10 breeds
        n += len(y)
    return correct / n


RESULTS_DIR = "results"


def save_result(name, acc):
    """Store a top-1 accuracy (fraction in [0, 1]) so `summary.py` can compare experiments."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump({"name": name, "acc": acc}, f)


# ── Position-embedding resizing (Author: Piotr) ─────────────


# This will return an enlarged pos_embed matrix, upsampling is done using bicubic interpolation
def enlarge_pos_embed(
    pos_embed: torch.Tensor, interpolation_mode="bicubic", new_grid=24
) -> torch.Tensor:
    # [CLS] and [DIST] tokens
    special_tokens = pos_embed[:, :2]
    pre_interpolaton = pos_embed[:, 2:]
    old_grid = isqrt(pre_interpolaton.shape[1])
    # Reshapes torch.Size([1, 196, 384]) to torch.Size([1, 384, 14, 14])

    pre_interpolaton = einops.rearrange(
        pre_interpolaton, "one (r c) D -> one D r c", r=old_grid
    )
    post_interpolation = F.interpolate(
        pre_interpolaton, (new_grid, new_grid), mode=interpolation_mode
    )

    post_interpolation = einops.rearrange(
        post_interpolation, "one D r c -> one (r c) D"
    )

    post_interpolation = torch.cat((special_tokens, post_interpolation), dim=1)
    return post_interpolation


@torch.no_grad()
def resize_pos_embed_(
    model: nn.Module, new_res=384, interpolation_mode="bicubic"
) -> nn.Module:
    """In place: resize the model's pos_embed for `new_res` inputs and update timm's patch_embed metadata."""
    new_grid = new_res // model.patch_embed.patch_size[0]

    model.pos_embed = nn.Parameter(
        enlarge_pos_embed(model.pos_embed.data, interpolation_mode, new_grid)
    )

    # timm asserts the input size against these, so they have to follow the new grid
    model.patch_embed.img_size = (new_res, new_res)
    model.patch_embed.grid_size = (new_grid, new_grid)
    model.patch_embed.num_patches = new_grid * new_grid
    return model

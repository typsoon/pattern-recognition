# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="SIE9HI1GVfK3"
# # **Task 4 — DINO**: break it, then fix it your way
#
# This week's seminar was about [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html). It has no labels and no negatives, so nothing in its loss forbids giving the same output to every image. Centering and sharpening stop this collapse, and the lecture showed how with schematic figures. Here you measure the real thing.
#
# This notebook gives you CIFAR-10, a tiny ViT, a **working DINO loop** and a **fixed evaluation**. The task is on the card.
#
# <details>
# <summary>Reach out here if you have any problems</summary>
# Taras Kutsyk — taras.kutsyk@doctoral.uj.edu.pl
# </details>

# %% [markdown] id="setup-info"
# ## Setup
# Run this notebook from the lab4 directory using a GPU-enabled environment.
#
# Implementation lives in `network.py`, `training.py`, `evaluation.py`,
# `common.py`, `checkpointing.py`, and `plots.py`. This notebook runs the experiments.

# %% id="imports"
from functools import partial

import matplotlib.pyplot as plt

from checkpointing import cached_train_dino
from common import load, normalize
from evaluation import knn_eval, present_baselines, present_the_outcome
from network import DINONetwork
from plots import plot_variants
from training import train_dino

# %% id="data"
# Load each split once; normalization is applied during training and evaluation.
train_images, train_labels = load("train")
test_images, test_labels = load("test")
print("train", train_images.shape, "test", test_images.shape)

# Bind datasets once, keeping experiment options and checkpoint names concise.
train = partial(train_dino, train_images, train_labels)
evaluate = partial(
    knn_eval,
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
)

# %% [markdown] id="baselines-info"
# ## Baselines
# Report the fixed raw-pixel and untrained-network k-NN baselines.
# Use `val_knn()` in `evaluation.py` for tuning, and the fixed test evaluation for final results.

# %% id="baselines"
present_baselines(train_images, train_labels, test_images, test_labels)

# %% [markdown] id="reference-info"
# ## Reference run
# Full DINO uses centering and sharpening. Completed runs load from `checkpoints/`.

# %% id="reference"
full = cached_train_dino(train, DINONetwork)
present_the_outcome(full, evaluate)

# %% [markdown] id="ablations-info"
# ## Part 1: remove centering and sharpening
# Each cell runs or loads one variant, then reports its student and teacher accuracy.

# %% id="no-center"
no_center = cached_train_dino(train, DINONetwork, center=False)
present_the_outcome(no_center, evaluate)

# %% id="no-sharpen"
no_sharpen = cached_train_dino(train, DINONetwork, sharpen=False)
present_the_outcome(no_sharpen, evaluate)

# %% id="neither"
no_center_and_no_sharpen = cached_train_dino(
    train, DINONetwork, center=False, sharpen=False
)
present_the_outcome(no_center_and_no_sharpen, evaluate)

# %% [markdown] id="plots-info"
# ## Compare all four variants
# The softmax stages use the same two test images. Training curves come from each run's history.

# %% id="plots"
variants = {
    "Full DINO": full,
    "No centering": no_center,
    "No sharpening": no_sharpen,
    "No centering or sharpening": no_center_and_no_sharpen,
}
targets_fig, training_fig = plot_variants(variants, normalize(test_images[:2]))

# Save each figure explicitly before displaying it.
targets_fig.savefig("teacher-softmax-stages.png", dpi=300, bbox_inches="tight")
training_fig.savefig("entropy-and-teacher-student-KL.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown] id="part2-info"
# ## Part 2: your own way to prevent collapse
# Implement your training changes in `training.py`, keeping the prescribed network,
# fixed evaluation, and budget of at most 20 epochs per run.

# %% id="part2"
# Part 2

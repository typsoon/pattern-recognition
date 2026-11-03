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

# %% [markdown] id="j3VUu56NVfK4"
# # Setup
#
# **Step 0.** Runtime → Change runtime type → **T4 GPU**. Everything we need is already installed on Colab.
#
# **Step 1. Data** (about 15 seconds). CIFAR-10: 50,000 training and 10,000 test images, 32×32, 10 classes, all kept on the GPU, so training does not wait for the CPU. We use a Hugging Face mirror because the original host is very slow. **The labels are used only inside `knn_eval()` and `val_knn()`.**

# %% id="JPmkWZnnVfK5"
import copy, io, time  # noqa: E401
from torch.autograd.grad_mode import _force_original_view_tracking
import numpy as np, pandas as pd, requests, torch, torch.nn as nn, torch.nn.functional as F  # noqa: E401
from PIL import Image
from pathlib import Path
import urllib.request
from common import load, normalize, device
from plots import plot_centering_sharpening, plot_curves, plot_variants
from checkpointing import cached_train_dino
import matplotlib.pyplot as plt


train_images, train_labels = load("train")
test_images, test_labels = load("test")


print("train", train_images.shape, "test", test_images.shape)


# %% [markdown] id="1HaNex7lVfK5"
# **Step 2. The network** (DON'T modify it). A ViT like those in Tasks 1 and 2, made small for 32×32 images: 4×4 patches, so 8×8 = 64 patches + the CLS token = 65 tokens; 4 blocks, 128 dimensions, 4 heads, 0.8M parameters. Smaller crops also work (the position embeddings are interpolated). Its output is the CLS token, **128 numbers: this is the feature k-NN uses**.


# %% id="myPo9oTeVfK6"
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(B, T, 3, self.heads, D // self.heads)
            .permute(2, 0, 3, 1, 4)  # qkv, B, H, T, D
        )
        x = x + self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        )  # residual connection
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, dim, 8, 8) * 0.02)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # (batch, 3, height, width), normalized -> (batch, 128) CLS features
        patches = self.patch_embed(x)
        pos = (
            self.pos
            if patches.shape[2:] == self.pos.shape[2:]
            else F.interpolate(
                self.pos, size=patches.shape[2:], mode="bicubic", align_corners=False
            )
        )
        x = (patches + pos).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1) + self.cls_pos, x], 1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


print("parameters:", sum(p.numel() for p in TinyViT().parameters()))


# %% [markdown] id="evRkrDS_VfK6"
# **Step 3. The fixed evaluation** (DON'T modify it). `knn_eval(backbone)` passes the images through your frozen network, stores the training features and labels, and classifies each test image by a weighted vote of its 20 nearest neighbours (about 3 seconds).
#
# - **`val_knn(backbone)`** does the same on a validation split: its queries are the last 5,000 training images, and its stored neighbours are the other 45,000 (the network still trains on all 50,000). **Choose your settings with `val_knn()`, and use `knn_eval()` only for your final numbers.**
# - **`collapse_curves(...)`** computes the numbers to log during training: the teacher's entropy and the teacher–student KL (the paper's Fig. 7), plus the entropy of the batch's average code.


# %% id="3Ndv1cazVfK7"
@torch.no_grad()
def features(backbone, images):
    # returns: (N, dim) L2-normalized features of the frozen network
    with torch.autocast("cuda", dtype=torch.float16):
        return F.normalize(
            torch.cat(
                [
                    backbone(normalize(images[i : i + 1024])).float()
                    for i in range(0, len(images), 1024)
                ]
            ),
            dim=-1,
        )


def weighted_knn(bank, bank_labels, queries, query_labels, k=20, temperature=0.07):
    # returns: top-1 accuracy of a similarity-weighted vote of the k nearest stored features
    correct = 0
    for i in range(0, len(queries), 1024):
        similarities, neighbours = (queries[i : i + 1024] @ bank.T).topk(k, dim=1)
        votes = torch.zeros(len(similarities), 10, device=device).scatter_add_(
            1, bank_labels[neighbours], (similarities / temperature).exp()
        )
        correct += (votes.argmax(1) == query_labels[i : i + 1024]).sum().item()
    return correct / len(queries)


def knn_eval(backbone):
    # The fixed evaluation for task 4. Do not change it; report exactly what it returns.
    # backbone - a network taking normalized images to (batch, dim) features
    # returns: top-1 accuracy on the CIFAR-10 test set
    was_training = backbone.training
    backbone.eval()
    accuracy = weighted_knn(
        features(backbone, train_images),
        train_labels,
        features(backbone, test_images),
        test_labels,
    )
    backbone.train(was_training)
    return accuracy


def val_knn(backbone):
    # The same as knn_eval, on the validation split. Use this one for every choice you tune.
    # returns: top-1 accuracy on the last 5,000 training images, with the other 45,000 as neighbours
    was_training = backbone.training
    backbone.eval()
    all_features = features(backbone, train_images)
    accuracy = weighted_knn(
        all_features[:45000],
        train_labels[:45000],
        all_features[45000:],
        train_labels[45000:],
    )
    backbone.train(was_training)
    return accuracy


@torch.no_grad()
def collapse_curves(teacher_probs, student_log_probs):
    # teacher_probs     - (batch, K) teacher distributions, after any centering and sharpening
    # student_log_probs - (batch, K) student log-probabilities, on the other view of the same images
    # returns: {"entropy": mean entropy of one image's code,
    #           "kl": mean KL(teacher || student),
    #           "batch_entropy": entropy of the batch's average code}
    # Entropies range from 0 (one-hot) to log K (uniform).
    log_teacher = teacher_probs.clamp_min(1e-12).log()
    mean = teacher_probs.mean(0)
    return {
        "entropy": -(teacher_probs * log_teacher).sum(-1).mean().item(),
        "kl": (teacher_probs * (log_teacher - student_log_probs)).sum(-1).mean().item(),
        "batch_entropy": -(mean * mean.clamp_min(1e-12).log()).sum().item(),
    }


# %% [markdown] id="a0RnfAAKVfK7"
# **Step 4. Baselines.** You should get about **45.7%** on raw pixels and **32.0%** with the untrained network. Both are far above the 10% of guessing, and raw pixels are the harder one to beat.

# %% id="FXaGJLsxVfK7"
pixels = lambda x: F.normalize(normalize(x).flatten(1), dim=-1)  # noqa: E731
print(
    "raw pixels, k-NN accuracy:",
    weighted_knn(pixels(train_images), train_labels, pixels(test_images), test_labels),
)
torch.manual_seed(0)
print("untrained network, k-NN accuracy:", knn_eval(TinyViT().to(device)))


# %% [markdown] id="lLQbhGVAVfK8"
# # The DINO loop
#
# **Step 5.** A compact version of the pseudocode from the lecture, with two views per image and no multi-crop. `center=False` turns centering off, and `sharpen=False` gives the teacher the student's temperature. `train_dino` returns the two networks, the final center and the logged curves, so you have what you need for the plots in Part 1.
#
# For Part 2, copy this cell and change it: everything here is yours to modify, except the network and the 20-epoch budget.


# %% id="g1zIMrY8VfK8"
def augment(images):
    # Two calls give two random views: crop-and-resize (30-100% of the area), flip, brightness and contrast, 20% greyscale.
    # images - (batch, 3, 32, 32) in [0, 1]; returns: the same shape, normalized
    n = len(images)
    scale = (0.3 + 0.7 * torch.rand(n, device=device)).sqrt()
    shift_x, shift_y = (
        (1 - scale) * (2 * torch.rand(n, device=device) - 1),
        (1 - scale) * (2 * torch.rand(n, device=device) - 1),
    )
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = (
        scale * flip,
        scale,
        shift_x,
        shift_y,
    )
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    views = F.grid_sample(images, grid, align_corners=False, padding_mode="reflection")
    brightness, contrast = (
        0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=device),
        0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=device),
    )
    views = (((views - 0.5) * contrast + 0.5) * brightness).clamp(0, 1)
    grey = torch.rand(n, 1, 1, 1, device=device) < 0.2
    return normalize(
        torch.where(grey, views.mean(1, keepdim=True).expand_as(views), views)
    )


class DINONetwork(nn.Module):
    # The backbone plus the head that produces the K codes: MLP, L2 normalization, weight-normalized linear layer.
    def __init__(self, K):
        super().__init__()
        self.backbone = TinyViT()
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.dim, 256), nn.GELU(), nn.Linear(256, 64)
        )
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(64, K, bias=False))

    def forward(self, x):
        return self.last(F.normalize(self.mlp(self.backbone(x)), dim=-1))


def train_dino(
    center=True,
    sharpen=True,
    epochs=20,
    seed=0,
    K=1024,
    batch_size=512,
    lr=1e-3,
    tau_s=0.1,
    tau_t=0.04,
    ema=0.996,
    center_momentum=0.9,
):
    # returns: {"student", "teacher", "center": the K-dim center, "tau_t", "history": one dict of curves per logged step}
    assert epochs <= 20, "the budget is 20 epochs per run"
    torch.manual_seed(seed)
    tau_t = tau_t if sharpen else tau_s
    student = DINONetwork(K).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, (step + 1) / 200)
    )
    scaler = torch.amp.GradScaler("cuda")
    c = torch.zeros(K, device=device)
    history, step, start = [], 0, time.time()

    def H(
        t, s
    ):  # cross-entropy with centered, sharpened teacher targets; the teacher gets no gradient
        return (
            -(F.softmax((t - c) / tau_t, -1) * F.log_softmax(s / tau_s, -1))
            .sum(-1)
            .mean()
        )

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_images), device=device)
        for i in range(len(order) // batch_size):
            x = train_images[order[i * batch_size : (i + 1) * batch_size]]
            x1, x2 = augment(x), augment(x)
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    t1, t2 = teacher(x1), teacher(x2)
                s1, s2 = student(x1), student(x2)
            t1, t2, s1, s2 = t1.float(), t2.float(), s1.float(), s2.float()
            loss = (H(t1, s2) + H(t2, s1)) / 2

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            warmup.step()  # noqa: E702

            step += 1
            if (
                step % 20 == 0 or i == len(order) // batch_size - 1
            ):  # logged with the same center the loss just used
                teacher_probs = F.softmax((t1 - c) / tau_t, -1)
                history.append(
                    dict(
                        step=step,
                        loss=loss.item(),
                        **collapse_curves(teacher_probs, F.log_softmax(s2 / tau_s, -1)),
                    )
                )

            with torch.no_grad():
                for p_t, p_s in zip(teacher.parameters(), student.parameters()):
                    p_t.mul_(ema).add_(p_s, alpha=1 - ema)
                if center:
                    c = center_momentum * c + (1 - center_momentum) * torch.cat(
                        [t1, t2]
                    ).mean(0)
        last = history[-1]
        val = f", val_knn {val_knn(student.backbone):.4f}" if epoch % 5 == 0 else ""
        print(
            f"epoch {epoch:2d} ({time.time() - start:4.0f}s): entropy {last['entropy']:.2f}, batch entropy {last['batch_entropy']:.2f}, KL {last['kl']:.3f}{val}",
            flush=True,
        )
    return dict(
        student=student, teacher=teacher, center=c, tau_t=tau_t, history=history
    )


# %% [markdown] id="rAHN5aAOVfK9"
# **Step 6. Reference run.** Full DINO, 20 epochs. You should get about **48.8%** with `knn_eval()` on the student's backbone. The largest possible entropy here is log 1024 ≈ 6.93.


# %% id="P_5meqPbVfK9"
def present_the_outcome(full, *, show_plots=True):
    print(
        "student k-NN:",
        knn_eval(full["student"].backbone),
        "| teacher k-NN:",
        knn_eval(full["teacher"].backbone),
    )  # type: ignore
    print("peak GPU memory, GB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))
    if show_plots:
        plot_curves(full)
        plot_centering_sharpening(full, normalize(test_images[:2]))
        plt.show()


full = cached_train_dino(train_dino, DINONetwork)
present_the_outcome(full, show_plots=False)

# %% [markdown] id="E_xxs_xJVfK9"
# # Your solution
#
# **Part 1:** `train_dino(center=False)`, `train_dino(sharpen=False)` and `train_dino(center=False, sharpen=False)`. **Part 2:** your own way to prevent collapse. Results vanish with the runtime, so save what you need to Drive. Good luck!
no_center = cached_train_dino(train_dino, DINONetwork, center=False)

present_the_outcome(no_center, show_plots=False)

# %%
no_sharpen = cached_train_dino(train_dino, DINONetwork, sharpen=False)

present_the_outcome(no_sharpen, show_plots=False)
# %%
no_center_and_no_sharpen = cached_train_dino(
    train_dino, DINONetwork, center=False, sharpen=False
)

present_the_outcome(no_center_and_no_sharpen, show_plots=False)
# %%
variants = {
    "Full DINO": full,
    "No centering": no_center,
    "No sharpening": no_sharpen,
    "No centering or sharpening": no_center_and_no_sharpen,
}
targets_fig, training_fig = plot_variants(variants, normalize(test_images[:2]))

targets_fig.savefig("teacher-softmax-stages.png", dpi=300, bbox_inches="tight")
training_fig.savefig("entropy-and-teacher-student-KL.png", dpi=300, bbox_inches="tight")
plt.show()

# %% id="BlYvBJpWVfK9"
# Part 1

# %% id="FrE0XMNDVfK-"
# Part 2

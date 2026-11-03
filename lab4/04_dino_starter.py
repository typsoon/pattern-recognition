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
import copy, time  # noqa: E401
import torch, torch.nn as nn, torch.nn.functional as F  # noqa: E401
from common import load, normalize, device
from plots import plot_variants
from checkpointing import cached_train_dino
import matplotlib.pyplot as plt


# Load unnormalized [0, 1] pixels once; augmentation and evaluation normalize them later.
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
        # Separate normalization precedes attention and the feed-forward sublayer.
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # One projection produces queries, keys and values; the other mixes attention heads.
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        # Expand each token's channels fourfold, apply GELU, then return to dim channels.
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, T, D = x.shape  # Batch size, token count and embedding dimension.
        q, k, v = (
            self.qkv(self.norm1(x))  # Normalize tokens and project all Q/K/V at once.
            .reshape(B, T, 3, self.heads, D // self.heads)  # Split Q/K/V and head dimensions.
            .permute(2, 0, 3, 1, 4)  # Arrange as (Q/K/V, batch, heads, tokens, head_dim).
        )
        # Attend within each head, concatenate head outputs, project, and add the input.
        x = x + self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        )  # residual connection
        return x + self.mlp(self.norm2(x))  # A second residual adds the token-wise MLP output.


class TinyViT(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)  # Embed non-overlapping image patches.
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))  # Learned token that summarizes the image.
        self.pos = nn.Parameter(torch.randn(1, dim, 8, 8) * 0.02)  # Learned patch-grid positions.
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))  # Separate position for the CLS token.
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))  # Register all blocks.
        self.norm = nn.LayerNorm(dim)  # Normalize embeddings before extracting CLS features.

    def forward(self, x):
        # (batch, 3, height, width), normalized -> (batch, 128) CLS features
        patches = self.patch_embed(x)  # Shape: (batch, dim, patch_rows, patch_columns).
        # Resize the learned position grid only when the input has a different patch grid.
        pos = (
            self.pos
            if patches.shape[2:] == self.pos.shape[2:]
            else F.interpolate(
                self.pos, size=patches.shape[2:], mode="bicubic", align_corners=False
            )
        )
        x = (patches + pos).flatten(2).transpose(1, 2)  # Convert the grid into a token sequence.
        # Prepend one copy of the shared CLS token per image; expand avoids copying its weights.
        x = torch.cat([self.cls.expand(len(x), -1, -1) + self.cls_pos, x], 1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]  # Return only the CLS token as each image's representation.


print("parameters:", sum(p.numel() for p in TinyViT().parameters()))


# %% [markdown] id="evRkrDS_VfK6"
# **Step 3. The fixed evaluation** (DON'T modify it). `knn_eval(backbone)` passes the images through your frozen network, stores the training features and labels, and classifies each test image by a weighted vote of its 20 nearest neighbours (about 3 seconds).
#
# - **`val_knn(backbone)`** does the same on a validation split: its queries are the last 5,000 training images, and its stored neighbours are the other 45,000 (the network still trains on all 50,000). **Choose your settings with `val_knn()`, and use `knn_eval()` only for your final numbers.**
# - **`collapse_curves(...)`** computes the numbers to log during training: the teacher's entropy and the teacher–student KL (the paper's Fig. 7), plus the entropy of the batch's average code.


# %% id="3Ndv1cazVfK7"
@torch.no_grad()  # Evaluation features do not require gradients or a training graph.
def features(backbone, images):
    # returns: (N, dim) L2-normalized features of the frozen network
    with torch.autocast("cuda", dtype=torch.float16):  # Reduce GPU work during feature extraction.
        # Unit-length features make their dot product equal to cosine similarity.
        return F.normalize(
            torch.cat(
                [
                    # Normalize pixels, extract a manageable batch, and retain float32 features.
                    backbone(normalize(images[i : i + 1024])).float()
                    for i in range(0, len(images), 1024)
                ]
            ),
            dim=-1,
        )


def weighted_knn(bank, bank_labels, queries, query_labels, k=20, temperature=0.07):
    # returns: top-1 accuracy of a similarity-weighted vote of the k nearest stored features
    correct = 0
    for i in range(0, len(queries), 1024):  # Batch queries to limit the similarity matrix size.
        # Compare each query with every bank feature and keep its k closest neighbours.
        similarities, neighbours = (queries[i : i + 1024] @ bank.T).topk(k, dim=1)
        # Accumulate exp(similarity / temperature) into each neighbour's class bin.
        votes = torch.zeros(len(similarities), 10, device=device).scatter_add_(
            1, bank_labels[neighbours], (similarities / temperature).exp()
        )
        correct += (votes.argmax(1) == query_labels[i : i + 1024]).sum().item()  # Count winning labels.
    return correct / len(queries)  # Convert the total count into top-1 accuracy.


def knn_eval(backbone):
    # The fixed evaluation for task 4. Do not change it; report exactly what it returns.
    # backbone - a network taking normalized images to (batch, dim) features
    # returns: top-1 accuracy on the CIFAR-10 test set
    was_training = backbone.training  # Preserve the mode used by the calling training loop.
    backbone.eval()  # Evaluate with training-specific layers disabled.
    # Use training images as the labelled bank and test images only as queries.
    accuracy = weighted_knn(
        features(backbone, train_images),
        train_labels,
        features(backbone, test_images),
        test_labels,
    )
    backbone.train(was_training)  # Resume the caller's original model mode.
    return accuracy


def val_knn(backbone):
    # The same as knn_eval, on the validation split. Use this one for every choice you tune.
    # returns: top-1 accuracy on the last 5,000 training images, with the other 45,000 as neighbours
    was_training = backbone.training  # Save the mode so validation does not alter training.
    backbone.eval()
    all_features = features(backbone, train_images)  # Extract the bank and queries in one pass.
    # Use disjoint parts of the training split, keeping the test set out of tuning.
    accuracy = weighted_knn(
        all_features[:45000],
        train_labels[:45000],
        all_features[45000:],
        train_labels[45000:],
    )
    backbone.train(was_training)  # Restore the mode after validation.
    return accuracy


@torch.no_grad()  # Logged diagnostics must not participate in gradient computation.
def collapse_curves(teacher_probs, student_log_probs):
    # teacher_probs     - (batch, K) teacher distributions, after any centering and sharpening
    # student_log_probs - (batch, K) student log-probabilities, on the other view of the same images
    # returns: {"entropy": mean entropy of one image's code,
    #           "kl": mean KL(teacher || student),
    #           "batch_entropy": entropy of the batch's average code}
    # Entropies range from 0 (one-hot) to log K (uniform).
    log_teacher = teacher_probs.clamp_min(1e-12).log()  # Avoid log(0) for tiny probabilities.
    mean = teacher_probs.mean(0)  # Average distributions across images, retaining K codes.
    return {
        # Sum -p*log(p) over codes, then average the per-image entropies.
        "entropy": -(teacher_probs * log_teacher).sum(-1).mean().item(),
        # Compare teacher targets with the student's predictions on the other view.
        "kl": (teacher_probs * (log_teacher - student_log_probs)).sum(-1).mean().item(),
        # Entropy of the batch average measures which codes are used across images.
        "batch_entropy": -(mean * mean.clamp_min(1e-12).log()).sum().item(),
    }


# %% [markdown] id="a0RnfAAKVfK7"
# **Step 4. Baselines.** You should get about **45.7%** on raw pixels and **32.0%** with the untrained network. Both are far above the 10% of guessing, and raw pixels are the harder one to beat.

# %% id="FXaGJLsxVfK7"
# Flatten standardized pixels and give each image unit length for a cosine k-NN baseline.
pixels = lambda x: F.normalize(normalize(x).flatten(1), dim=-1)  # noqa: E731
print(
    "raw pixels, k-NN accuracy:",
    weighted_knn(pixels(train_images), train_labels, pixels(test_images), test_labels),
)
torch.manual_seed(0)  # Make the untrained-network baseline reproducible.
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
    # Sample crop area uniformly from [0.3, 1]; square root converts area to side length.
    scale = (0.3 + 0.7 * torch.rand(n, device=device)).sqrt()
    # Move the crop within the image's normalized [-1, 1] coordinate system.
    shift_x, shift_y = (
        (1 - scale) * (2 * torch.rand(n, device=device) - 1),
        (1 - scale) * (2 * torch.rand(n, device=device) - 1),
    )
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)  # Flip horizontally half the time.
    theta = torch.zeros(n, 2, 3, device=device)  # One affine sampling transform per image.
    # Encode horizontal flip, crop size and crop position in the affine matrices.
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = (
        scale * flip,
        scale,
        shift_x,
        shift_y,
    )
    grid = F.affine_grid(theta, images.shape, align_corners=False)  # Coordinates to sample per pixel.
    views = F.grid_sample(images, grid, align_corners=False, padding_mode="reflection")  # Crop and resize.
    # Independent per-image brightness and contrast factors, each in [0.6, 1.4].
    brightness, contrast = (
        0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=device),
        0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=device),
    )
    views = (((views - 0.5) * contrast + 0.5) * brightness).clamp(0, 1)  # Adjust and bound pixels.
    grey = torch.rand(n, 1, 1, 1, device=device) < 0.2  # Choose 20% of images for greyscale.
    # Broadcast the greyscale channel average back to RGB, then standardize channels.
    return normalize(
        torch.where(grey, views.mean(1, keepdim=True).expand_as(views), views)
    )


class DINONetwork(nn.Module):
    # The backbone plus the head that produces the K codes: MLP, L2 normalization, weight-normalized linear layer.
    def __init__(self, K):
        super().__init__()
        self.backbone = TinyViT()  # The backbone's CLS feature is used for downstream k-NN.
        # Map backbone features into the smaller space used by the projection head.
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.dim, 256), nn.GELU(), nn.Linear(256, 64)
        )
        # Produce K code logits with separately parameterized weight directions and magnitudes.
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(64, K, bias=False))

    def forward(self, x):
        # Extract features, project them, normalize their length, then compute code logits.
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
    assert epochs <= 20, "the budget is 20 epochs per run"  # Enforce the lab's training budget.
    torch.manual_seed(seed)  # Reproduce initialization, batch order and random augmentations.
    tau_t = tau_t if sharpen else tau_s  # Disable sharpening by matching student temperature.
    student = DINONetwork(K).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False)  # Identical initial weights, no teacher gradients.
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)  # Optimize the student only.
    # Ramp the learning rate up to lr over the first 200 update steps.
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, (step + 1) / 200)
    )
    scaler = torch.amp.GradScaler("cuda")  # Scale gradients to reduce float16 underflow.
    c = torch.zeros(K, device=device)  # Running mean of teacher logits, initially neutral.
    history, step, start = [], 0, time.time()  # Store logged metrics, update count and elapsed time.

    def H(
        t, s
    ):  # cross-entropy with centered, sharpened teacher targets; the teacher gets no gradient
        # Teacher probabilities weight the student's negative log-probabilities.
        return (
            -(F.softmax((t - c) / tau_t, -1) * F.log_softmax(s / tau_s, -1))
            .sum(-1)  # Sum cross-entropy over K codes for each image.
            .mean()  # Average the per-image losses across the batch.
        )

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_images), device=device)  # Reshuffle images each epoch.
        for i in range(len(order) // batch_size):  # Use complete batches; skip the final remainder.
            # Slice shuffled indices so each training image appears at most once this epoch.
            x = train_images[order[i * batch_size : (i + 1) * batch_size]]
            x1, x2 = augment(x), augment(x)  # Independent augmented views of the same images.
            with torch.autocast("cuda", dtype=torch.float16):  # Use mixed precision for forward passes.
                with torch.no_grad():  # Teacher outputs act as fixed targets for this update.
                    t1, t2 = teacher(x1), teacher(x2)
                s1, s2 = student(x1), student(x2)
            t1, t2, s1, s2 = t1.float(), t2.float(), s1.float(), s2.float()  # Stable float32 loss math.
            loss = (H(t1, s2) + H(t2, s1)) / 2  # Match opposite views in both directions.

            optimizer.zero_grad()  # Discard gradients from the previous batch.
            scaler.scale(loss).backward()  # Backpropagate a scaled loss to protect small gradients.
            scaler.unscale_(optimizer)  # Restore true gradient magnitudes before clipping.
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)  # Limit the total gradient norm.
            scaler.step(optimizer)  # Update the student; skip the update if gradients overflow.
            scaler.update()  # Adapt the scale for the next batch.
            warmup.step()  # Advance the learning-rate warmup schedule.

            step += 1
            if (
                # Log regularly and at the end of each epoch, including uneven intervals.
                step % 20 == 0 or i == len(order) // batch_size - 1
            ):  # logged with the same center the loss just used
                teacher_probs = F.softmax((t1 - c) / tau_t, -1)  # Reuse the loss's center and logits.
                history.append(
                    dict(
                        step=step,
                        loss=loss.item(),  # Store a plain number rather than a graph-connected tensor.
                        # Expand the metric dictionary into this step's history entry.
                        **collapse_curves(teacher_probs, F.log_softmax(s2 / tau_s, -1)),
                    )
                )

            with torch.no_grad():  # Teacher and center updates are moving averages, not optimization.
                # Match each teacher parameter to the corresponding student parameter.
                for p_t, p_s in zip(teacher.parameters(), student.parameters()):
                    p_t.mul_(ema).add_(p_s, alpha=1 - ema)  # teacher = ema*teacher + (1-ema)*student.
                if center:  # Otherwise c stays zero throughout training.
                    # Average logits over both views and update the running per-code mean.
                    c = center_momentum * c + (1 - center_momentum) * torch.cat(
                        [t1, t2]
                    ).mean(0)
        last = history[-1]  # Report the final logged batch from this epoch.
        val = f", val_knn {val_knn(student.backbone):.4f}" if epoch % 5 == 0 else ""  # Validate every 5 epochs.
        print(
            f"epoch {epoch:2d} ({time.time() - start:4.0f}s): entropy {last['entropy']:.2f}, batch entropy {last['batch_entropy']:.2f}, KL {last['kl']:.3f}{val}",
            flush=True,  # Display progress immediately in notebooks or redirected logs.
        )
    return dict(
        student=student, teacher=teacher, center=c, tau_t=tau_t, history=history
    )


# %% [markdown] id="rAHN5aAOVfK9"
# **Step 6. Reference run.** Full DINO, 20 epochs. You should get about **48.8%** with `knn_eval()` on the student's backbone. The largest possible entropy here is log 1024 ≈ 6.93.


# %% id="P_5meqPbVfK9"
def present_the_outcome(full):
    # Compare student and EMA-teacher features with the fixed test-set evaluator.
    print(
        "student k-NN:",
        knn_eval(full["student"].backbone),
        "| teacher k-NN:",
        knn_eval(full["teacher"].backbone),
    )  # type: ignore
    # Convert PyTorch's process-wide peak allocated bytes into GiB for reporting.
    print("peak GPU memory, GB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))


full = cached_train_dino(train_dino, DINONetwork)  # Load or train the centering + sharpening baseline.
present_the_outcome(full)

# %% [markdown] id="E_xxs_xJVfK9"
# # Your solution
#
# **Part 1:** `train_dino(center=False)`, `train_dino(sharpen=False)` and `train_dino(center=False, sharpen=False)`. **Part 2:** your own way to prevent collapse. Results vanish with the runtime, so save what you need to Drive. Good luck!
no_center = cached_train_dino(train_dino, DINONetwork, center=False)  # Keep sharpening, disable centering.

present_the_outcome(no_center)

# %%
no_sharpen = cached_train_dino(train_dino, DINONetwork, sharpen=False)  # Keep centering, equalize temperatures.

present_the_outcome(no_sharpen)
# %%
# Disable both operations to measure agreement without either collapse-prevention step.
no_center_and_no_sharpen = cached_train_dino(
    train_dino, DINONetwork, center=False, sharpen=False
)

present_the_outcome(no_center_and_no_sharpen)
# %%
# Preserve this order for the subplot rows, line colors and legends.
variants = {
    "Full DINO": full,
    "No centering": no_center,
    "No sharpening": no_sharpen,
    "No centering or sharpening": no_center_and_no_sharpen,
}
# Use the same two standardized test images for every teacher, returning separate figure handles.
targets_fig, training_fig = plot_variants(variants, normalize(test_images[:2]))

# Save through explicit figure handles so each filename receives the intended figure.
targets_fig.savefig("teacher-softmax-stages.png", dpi=300, bbox_inches="tight")
training_fig.savefig("entropy-and-teacher-student-KL.png", dpi=300, bbox_inches="tight")
plt.show()

# %% id="BlYvBJpWVfK9"
# Part 1

# %% id="FrE0XMNDVfK-"
# Part 2

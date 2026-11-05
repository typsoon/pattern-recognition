"""DINO augmentations, collapse diagnostics, and training loop."""


import copy
import time

import torch
from torch import nn
from torch.nn import functional as F

from common import device, normalize
from evaluation import val_knn
from network import DINONetwork


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


def train_dino(
    train_images,
    train_labels,
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
    extra_loss=None,
    target_transform=None,
):
    """Train a DINO student against an EMA teacher using two augmented views.

    Each teacher target is matched to the student's prediction on the opposite
    view. Teacher targets are computed without gradients; only the student is
    optimized. Validation k-NN is reported every five epochs.

    Args:
        extra_loss: Optional callable ``(s1, s2) -> scalar`` added to the
            cross-view loss. Inputs are float32 student logits of shape
            ``(batch_size, K)``; the returned loss can carry student gradients.
        target_transform: Optional callable ``(t1, t2) -> (p1, p2)`` receiving
            float32 teacher logits and returning target probabilities, all of
            shape ``(batch_size, K)``. Called once per batch without gradients.
            By default, targets are ``softmax((t - c) / tau_t)``. A custom
            transform handles its own centering, scaling, and temperature.
        center: Whether to update the running teacher-logit center. If false,
            the center stays zero. Use false for batch-standardized targets.
        sharpen: Whether to keep the supplied teacher temperature. If false,
            ``tau_t`` is set to ``tau_s``. Custom targets bypass this temperature.

    Returns:
        A dictionary containing ``student``, ``teacher``, the final ``center``
        of shape ``(K,)``, the effective ``tau_t``, and ``history``. History entries
        contain the step, total loss, teacher entropy, batch entropy, and KL
        divergence to the student's prediction on the opposite view.
    """
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

    def H(target, log_student):
        """Sum cross-entropy over K codes and average over the batch.

        Both inputs have shape ``(batch_size, K)``: ``target`` contains teacher
        probabilities and ``log_student`` contains student log-probabilities.
        Return a scalar loss with gradients through the student prediction.
        """
        return -(target * log_student).sum(-1).mean()

    def default_transform(t1, t2):
        """Return centered, temperature-scaled probabilities for both teacher views.

        Accept two teacher-logit tensors of shape ``(batch_size, K)`` and return
        probability tensors of the same shape. Use the enclosing scope's current
        running center ``c`` and effective teacher temperature ``tau_t``.
        """
        return F.softmax((t1 - c) / tau_t, -1), F.softmax((t2 - c) / tau_t, -1)

    target_transform = default_transform if target_transform is None else target_transform

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
            with torch.no_grad():
                target1, target2 = target_transform(t1, t2)

            log_student1 = F.log_softmax(s1 / tau_s, -1)
            log_student2 = F.log_softmax(s2 / tau_s, -1)
            # Match each teacher target with the student's prediction on the opposite view.
            loss = (H(target1, log_student2) + H(target2, log_student1)) / 2
            if extra_loss is not None:  # Part 2 regularisers, e.g. fix_mutual_info.py.
                loss = loss + extra_loss(s1, s2)

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
                history.append(
                    dict(
                        step=step,
                        loss=loss.item(),  # Store a plain number rather than a graph-connected tensor.
                        # Expand the metric dictionary into this step's history entry.
                        **collapse_curves(target1, log_student2),
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
        val = f", val_knn {val_knn(student.backbone, train_images, train_labels):.4f}" if epoch % 5 == 0 else ""  # Validate every 5 epochs.
        print(
            f"epoch {epoch:2d} ({time.time() - start:4.0f}s): entropy {last['entropy']:.2f}, batch entropy {last['batch_entropy']:.2f}, KL {last['kl']:.3f}{val}",
            flush=True,  # Display progress immediately in notebooks or redirected logs.
        )
    return dict(
        student=student, teacher=teacher, center=c, tau_t=tau_t, history=history
    )

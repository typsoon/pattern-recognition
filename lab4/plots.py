"""Plot training diagnostics and the teacher's centering and sharpening."""

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(result, *, label=None, axes=None):
    """Plot teacher entropy and KL(teacher || student) against training steps.

    Pass the dictionary returned by ``train_dino``. Reuse the returned axes
    and supply labels to compare runs::

        fig, axes = plot_curves(full, label="Full DINO")
        plot_curves(no_center, label="No centering", axes=axes)
        plt.show()

    Returns the Matplotlib figure and its two axes.
    """
    history = result["history"]
    if not history:
        raise ValueError("Training history is empty; there are no curves to plot.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    else:
        fig = axes[0].figure

    steps = [entry["step"] for entry in history]
    for ax, key, title, ylabel in zip(
        axes,
        ("entropy", "kl"),
        ("Teacher entropy", "Teacher–student KL"),
        ("Entropy (nats)", "KL(teacher || student) (nats)"),
    ):
        ax.plot(steps, [entry[key] for entry in history], label=label)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        if label is not None:
            ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_centering_sharpening(result, images, *, max_codes=12):
    """Show logits -> centering -> softmax for two normalized input images.

    ``result`` is returned by ``train_dino``; ``images`` has shape (2, 3, H, W)
    and must already be normalized and on the teacher's device. For example::

        plot_centering_sharpening(full, normalize(train_images[:2]))
        plt.show()

    Display the codes with the largest target probability in either image,
    retaining their original probabilities without renormalizing the subset.
    Entropy and total variation distance use all K codes. A run without
    centering already has a zero center in its result.

    Returns the Matplotlib figure and its four axes.
    """
    import torch

    if len(images) != 2:
        raise ValueError("Provide exactly two normalized images to compare.")
    if not isinstance(max_codes, int) or max_codes < 1:
        raise ValueError("max_codes must be a positive integer.")

    teacher = result["teacher"]
    was_training = teacher.training
    teacher.eval()
    try:
        with torch.no_grad():
            logits = teacher(images).float().cpu().numpy()
            center = result["center"].detach().float().cpu().numpy()
    finally:
        teacher.train(was_training)

    centered = logits - center
    scaled = centered / result["tau_t"]
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    entropy = -(probs * log_probs).sum(axis=-1)
    distance = 0.5 * np.abs(probs[0] - probs[1]).sum()
    codes = np.sort(np.argsort(probs.max(axis=0))[-max_codes:])
    code_labels = codes.tolist()
    values = [logits[:, codes], center[codes], centered[:, codes], probs[:, codes]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    x = np.arange(len(code_labels))
    colors = ("#6236b5", "#198587")
    titles = (
        r"Teacher logits $g_t(x)$",
        r"Center $c$",
        r"Centered logits $g_t(x) - c$",
        r"Teacher target: softmax$((g_t(x)-c)/\tau_t)$" + "\n"
        + rf"$\tau_t = {result['tau_t']:g}$",
    )
    for i, (ax, data, title) in enumerate(zip(axes, values, titles)):
        if i == 1:
            ax.bar(x, data, width=0.7, color="#d99a10")
        else:
            for row, (label, color) in enumerate(zip(("Image A", "Image B"), colors)):
                ax.bar(x + (row - 0.5) * 0.36, data[row], width=0.36,
                       color=color, label=label)
        ax.set(title=title, xlabel="Code index")
        ax.set_xticks(x, code_labels, rotation=90)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Logit")
    axes[0].legend()
    axes[3].set(ylabel="Probability", ylim=(0, 1))
    fig.suptitle("What the teacher's softmax sees: centering and sharpening")
    fig.text(
        0.5, 0.025,
        f"Entropy (nats): A {entropy[0]:.2f}, B {entropy[1]:.2f}"
        f"   |   Total variation distance: {distance:.2f}"
        f"   |   Showing {len(code_labels)} of {logits.shape[-1]} codes",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2)
    return fig, axes

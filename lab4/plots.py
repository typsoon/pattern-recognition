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

        plot_centering_sharpening(full, normalize(test_images[:2]))
        plt.show()

    Display the codes with the largest target probability in either image,
    retaining their original probabilities without renormalizing the subset.
    Entropy and total variation distance use all K codes. A run without
    centering already has a zero center in its result.

    Returns the Matplotlib figure and its four axes.
    """
    data = _teacher_targets(result, images)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    shown = _plot_target_stages(data, axes, max_codes=max_codes)
    fig.suptitle("What the teacher's softmax sees: centering and sharpening")
    fig.text(
        0.5, 0.025,
        f"Entropy (nats): A {data['entropy'][0]:.2f}, B {data['entropy'][1]:.2f}"
        f"   |   Total variation distance: {data['distance']:.2f}"
        f"   |   Showing {shown} of {data['logits'].shape[-1]} codes",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2)
    return fig, axes


def plot_variants(results, images, *, max_codes=12):
    """Return two figures comparing named DINO runs on the same two test images.

    ``results`` maps variant names to dictionaries returned by ``train_dino``.
    ``images`` contains exactly two normalized test images on the teachers'
    device. The first figure has one row per variant and four columns:
    logits, center, centered logits, and teacher probabilities. The second
    overlays every variant's logged entropy and KL on two training panels.

    Each run displays its own most active codes (indices do not have shared
    meanings across separately trained networks). Probabilities are computed
    over all K codes before selecting bars. Entropy and total variation also
    use all K codes; the caption reports the probability mass shown.

    Example::

        runs = {"Full DINO": full, "No centering": no_center,
                "No sharpening": no_sharpen,
                "Neither": no_center_and_no_sharpen}
        targets_fig, training_fig = plot_variants(
            runs, normalize(test_images[:2]))
        plt.show()
    """
    if not results:
        raise ValueError("Provide at least one training result to compare.")
    if any(not result["history"] for result in results.values()):
        raise ValueError("Every variant must have a nonempty training history.")
    if not isinstance(max_codes, int) or max_codes < 1:
        raise ValueError("max_codes must be a positive integer.")

    # Finish inference before creating figures, preserving every model's mode.
    targets = [(label, _teacher_targets(result, images))
               for label, result in results.items()]
    targets_fig, target_axes = plt.subplots(
        len(targets), 4, figsize=(18, 3.8 * len(targets)), squeeze=False,
    )
    targets_fig.suptitle(
        "What the teacher's softmax actually sees\n"
        "Same two test images for every variant; entropy and distance use all codes",
        fontsize=15,
    )
    for axes, (label, data) in zip(target_axes, targets):
        shown = _plot_target_stages(data, axes, max_codes=max_codes)
        axes[0].set_title(f"{label}\n" + axes[0].get_title(), fontweight="bold")
        axes[3].set_xlabel(
            f"{shown}/{data['logits'].shape[-1]} codes; "
            f"shown mass A {data['shown_mass'][0]:.1%}, B {data['shown_mass'][1]:.1%}\n"
            f"Entropy (nats): A {data['entropy'][0]:.2f}, B {data['entropy'][1]:.2f}\n"
            f"Total variation distance: {data['distance']:.2f}",
            fontsize=9,
        )
    targets_fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2)

    curves_fig, curve_axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for label, result in results.items():
        plot_curves(result, label=label, axes=curve_axes)
    curves_fig.suptitle("Training diagnostics — all variants")
    curves_fig.tight_layout(rect=(0, 0, 1, 0.93))
    return targets_fig, curves_fig


def _teacher_targets(result, images):
    """Evaluate a teacher and compute its full distributions without gradients."""
    import torch

    if len(images) != 2:
        raise ValueError("Provide exactly two normalized test images to compare.")
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
    return dict(
        logits=logits, center=center, centered=centered, probs=probs,
        entropy=-(probs * log_probs).sum(axis=-1),
        distance=0.5 * np.abs(probs[0] - probs[1]).sum(), tau_t=result["tau_t"],
    )


def _plot_target_stages(data, axes, *, max_codes):
    """Draw one variant using the same selected code indices in every panel."""
    if not isinstance(max_codes, int) or max_codes < 1:
        raise ValueError("max_codes must be a positive integer.")
    codes = np.sort(np.argsort(data["probs"].max(axis=0))[-max_codes:])
    code_labels = codes.tolist()
    values = [data["logits"][:, codes], data["center"][codes],
              data["centered"][:, codes], data["probs"][:, codes]]
    data["shown_mass"] = data["probs"][:, codes].sum(axis=-1)
    x = np.arange(len(code_labels))
    colors = ("#6236b5", "#198587")
    titles = (
        r"Teacher logits $g_t(x)$",
        r"Center $c$",
        r"Centered logits $g_t(x) - c$",
        r"Teacher target: softmax$((g_t(x)-c)/\tau_t)$" + "\n"
        + rf"$\tau_t = {data['tau_t']:g}$",
    )
    for i, (ax, bars, title) in enumerate(zip(axes, values, titles)):
        if i == 1:
            ax.bar(x, bars, width=0.7, color="#d99a10")
        else:
            for row, (label, color) in enumerate(zip(("Image A", "Image B"), colors)):
                ax.bar(x + (row - 0.5) * 0.36, bars[row], width=0.36,
                       color=color, label=label)
        ax.set(title=title, xlabel="Code index")
        ax.set_xticks(x, code_labels, rotation=90)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Logit")
    axes[0].legend()
    axes[3].set(ylabel="Probability", ylim=(0, 1))
    return len(codes)

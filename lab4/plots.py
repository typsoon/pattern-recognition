"""Plot training diagnostics and the teacher's centering and sharpening."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_variants(results, images, *, max_codes=12):
    """Return (softmax_stages_figure, training_curves_figure) for named runs.

    Pass a mapping of names to train_dino results and two normalized test
    images on the teachers' device. Each run displays its own most active
    codes without renormalizing their probabilities; entropy and distance
    use all K codes. Code indices are not aligned across trained networks.
    """
    if not results:
        raise ValueError("Provide at least one training result to compare.")
    if any(not result["history"] for result in results.values()):
        raise ValueError("Every variant must have a nonempty training history.")
    if not isinstance(max_codes, int) or max_codes < 1:
        raise ValueError("max_codes must be a positive integer.")
    if len(images) != 2:
        raise ValueError("Provide exactly two normalized test images to compare.")

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
        codes = _plot_target_stages(data, axes, max_codes=max_codes)
        shown_mass = data["probs"][:, codes].sum(axis=-1)
        axes[0].set_title(f"{label}\n" + axes[0].get_title(), fontweight="bold")
        axes[3].set_xlabel(
            f"{len(codes)}/{data['logits'].shape[-1]} codes; "
            f"shown mass A {shown_mass[0]:.1%}, B {shown_mass[1]:.1%}\n"
            f"Entropy (nats): A {data['entropy'][0]:.2f}, B {data['entropy'][1]:.2f}\n"
            f"Total variation distance: {data['distance']:.2f}",
            fontsize=9,
        )
    targets_fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2)

    curves_fig, curve_axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, key, title, ylabel in zip(
        curve_axes,
        ("entropy", "kl"),
        ("Teacher entropy", "Teacher–student KL"),
        ("Entropy (nats)", "KL(teacher || student) (nats)"),
    ):
        for label, result in results.items():
            history = result["history"]
            ax.plot([row["step"] for row in history],
                    [row[key] for row in history], label=label)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    curves_fig.suptitle("Training diagnostics — all variants")
    curves_fig.tight_layout(rect=(0, 0, 1, 0.93))
    return targets_fig, curves_fig


@torch.no_grad()
def _teacher_targets(result, images):
    """Evaluate a teacher and compute its full distributions without gradients."""
    teacher = result["teacher"]
    was_training = teacher.training
    teacher.eval()
    try:
        logits = teacher(images).float()
    finally:
        teacher.train(was_training)

    center = result["center"].float()
    centered = logits - center
    log_probs = torch.log_softmax(centered / result["tau_t"], dim=-1)
    probs = log_probs.exp()
    arrays = {name: value.cpu().numpy() for name, value in dict(
        logits=logits, center=center, centered=centered, probs=probs,
        entropy=-(probs * log_probs).sum(dim=-1),
    ).items()}
    return dict(
        **arrays, distance=0.5 * (probs[0] - probs[1]).abs().sum().item(),
        tau_t=result["tau_t"],
    )


def _plot_target_stages(data, axes, *, max_codes):
    """Draw one variant using the same selected code indices in every panel."""
    codes = np.sort(np.argsort(data["probs"].max(axis=0))[-max_codes:])
    code_labels = codes.tolist()
    values = [data["logits"][:, codes], data["center"][codes],
              data["centered"][:, codes], data["probs"][:, codes]]
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
    return codes

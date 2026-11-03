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
    # Reject empty histories before creating figures that cannot show training.
    if any(not result["history"] for result in results.values()):
        raise ValueError("Every variant must have a nonempty training history.")
    # A positive integer controls how many of the K output codes are visible.
    if not isinstance(max_codes, int) or max_codes < 1:
        raise ValueError("max_codes must be a positive integer.")
    # Every variant must compare the same pair of images, labeled A and B.
    if len(images) != 2:
        raise ValueError("Provide exactly two normalized test images to compare.")

    # Finish inference before creating figures, preserving every model's mode.
    targets = [(label, _teacher_targets(result, images))
               for label, result in results.items()]
    # Each row is one variant; columns show the four stages from the lecture.
    targets_fig, target_axes = plt.subplots(
        # Keep axes two-dimensional even when results contains only one run.
        len(targets), 4, figsize=(18, 3.8 * len(targets)), squeeze=False,
    )
    targets_fig.suptitle(
        "What the teacher's softmax actually sees\n"
        "Same two test images for every variant; entropy and distance use all codes",
        fontsize=15,
    )
    # Pair each subplot row with the corresponding variant's computed values.
    for axes, (label, data) in zip(target_axes, targets):
        # Reuse the selected code indices to describe what the bars omit.
        codes = _plot_target_stages(data, axes, max_codes=max_codes)
        # Sum displayed probabilities for each image without renormalizing them.
        shown_mass = data["probs"][:, codes].sum(axis=-1)
        # Put the variant name above the first panel of its row.
        axes[0].set_title(f"{label}\n" + axes[0].get_title(), fontweight="bold")
        # Report full-distribution diagnostics below the selected probability bars.
        axes[3].set_xlabel(
            f"{len(codes)}/{data['logits'].shape[-1]} codes; "
            f"shown mass A {shown_mass[0]:.1%}, B {shown_mass[1]:.1%}\n"
            f"Entropy (nats): A {data['entropy'][0]:.2f}, B {data['entropy'][1]:.2f}\n"
            f"Total variation distance: {data['distance']:.2f}",
            fontsize=9,
        )
    # Reserve the top 5% for the figure title and separate rows and columns.
    targets_fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2)

    # Both diagnostics use the same training-step axis, with independent y scales.
    curves_fig, curve_axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    # Bind each metric key to its subplot title and units.
    for ax, key, title, ylabel in zip(
        curve_axes,
        ("entropy", "kl"),
        ("Teacher entropy", "Teacher–student KL"),
        ("Entropy (nats)", "KL(teacher || student) (nats)"),
    ):
        for label, result in results.items():
            history = result["history"]
            # Use actual logged steps, which need not be evenly spaced.
            ax.plot([row["step"] for row in history],
                    [row[key] for row in history], label=label)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.grid(True, alpha=0.3)  # A faint grid helps compare the four curves.
        ax.legend()  # Match line colors to variant names in each panel.
    curves_fig.suptitle("Training diagnostics — all variants")
    curves_fig.tight_layout(rect=(0, 0, 1, 0.93))  # Leave room for the main title.
    return targets_fig, curves_fig  # Explicit handles let callers save both figures.


@torch.no_grad()  # Inference and diagnostics must not build an autograd graph.
def _teacher_targets(result, images):
    """Evaluate a teacher and compute its full distributions without gradients."""
    teacher = result["teacher"]
    was_training = teacher.training  # Remember the caller's original model mode.
    teacher.eval()  # Disable any training-specific behavior during measurement.
    try:
        # Compute all K logits for both images in float32 before softmax.
        logits = teacher(images).float()
    finally:
        # Restore the mode even if the forward pass raises an exception.
        teacher.train(was_training)

    # Use the trained running center and the temperature actually used by this run.
    center = result["center"].float()
    centered = logits - center  # Broadcast the K-dimensional center over both images.
    # Stable log-softmax normalizes over all codes; a lower temperature sharpens it.
    log_probs = torch.log_softmax(centered / result["tau_t"], dim=-1)
    probs = log_probs.exp()  # Convert log-probabilities into bar heights.
    # Matplotlib consumes CPU arrays; convert only after computing full distributions.
    arrays = {name: value.cpu().numpy() for name, value in dict(
        logits=logits, center=center, centered=centered, probs=probs,
        # Sum -p*log(p) over codes to obtain one entropy per image, in nats.
        entropy=-(probs * log_probs).sum(dim=-1),
    ).items()}
    return dict(
        # Total variation compares the two teacher distributions: 0=same, 1=disjoint.
        **arrays, distance=0.5 * (probs[0] - probs[1]).abs().sum().item(),
        tau_t=result["tau_t"],
    )


def _plot_target_stages(data, axes, *, max_codes):
    """Draw one variant using the same selected code indices in every panel."""
    # Rank codes by their strongest probability in either image, keep the top few,
    # then restore code-index order so all four panels line up.
    codes = np.sort(np.argsort(data["probs"].max(axis=0))[-max_codes:])
    code_labels = codes.tolist()  # Display original indices, not their ranking positions.
    # Select the same columns from logits, center, centered logits and probabilities.
    values = [data["logits"][:, codes], data["center"][codes],
              data["centered"][:, codes], data["probs"][:, codes]]
    x = np.arange(len(code_labels))  # Equally spaced positions for the selected codes.
    colors = ("#6236b5", "#198587")  # Purple is image A; teal is image B in every row.
    titles = (
        r"Teacher logits $g_t(x)$",
        r"Center $c$",
        r"Centered logits $g_t(x) - c$",
        r"Teacher target: softmax$((g_t(x)-c)/\tau_t)$" + "\n"
        + rf"$\tau_t = {data['tau_t']:g}$",
    )
    # Associate each stage's values and title with its subplot.
    for i, (ax, bars, title) in enumerate(zip(axes, values, titles)):
        if i == 1:  # The center is shared by both images, so it needs one bar per code.
            ax.bar(x, bars, width=0.7, color="#d99a10")
        else:
            # Other stages have two values per code, one for each image.
            for row, (label, color) in enumerate(zip(("Image A", "Image B"), colors)):
                # Offset A and B equally left/right to draw adjacent bars.
                ax.bar(x + (row - 0.5) * 0.36, bars[row], width=0.36,
                       color=color, label=label)
        ax.set(title=title, xlabel="Code index")
        ax.set_xticks(x, code_labels, rotation=90)  # Vertical labels fit large code indices.
        ax.axhline(0, color="gray", linewidth=0.6)  # Separate positive and negative logits.
        ax.grid(axis="y", alpha=0.2)  # Compare bar heights without vertical grid clutter.
        ax.set_axisbelow(True)  # Keep the grid behind the bars.

    axes[0].set_ylabel("Logit")
    axes[0].legend()
    axes[3].set(ylabel="Probability", ylim=(0, 1))  # Use the same scale for every variant.
    return codes  # The caller uses these indices to calculate the displayed mass.

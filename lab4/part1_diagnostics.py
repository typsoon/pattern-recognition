"""TODO — Person 1: implement the Part 1 DINO diagnostics runner.

Required work:
* Run all four center/sharpen combinations for 20 epochs with seed 0.
* Run the full-DINO baseline with seeds 1 and 2; retain the raw-pixel and
  untrained-TinyViT k-NN baselines.
* Record the required validation/test accuracies and training histories in
  the shared results/artifact format.
* Produce the three-panel entropy, marginal-entropy, and KL plot; produce the
  four-stage teacher-logit/softmax figure on two test images; write a
  quantitative notes.md based on actual run outputs.

Constraints: do not use labels in training, exceed 20 epochs per run, or edit
the original ``04_dino_starter.ipynb`` or the fixed evaluation implementation.
"""


def main():
    raise NotImplementedError(
        "Person 1 must implement the Part 1 diagnostics described in this file."
    )


if __name__ == "__main__":
    main()

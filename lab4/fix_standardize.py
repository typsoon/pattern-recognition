"""TODO — Person 2: implement the target-side standardization experiment.

Required work:
* Replace teacher centering and temperature sharpening with batch
  standardization; use equal teacher/student temperatures and no center.
* Implement ``standardized_targets`` over concatenated teacher logits from
  both augmented views. Support scalar standard deviation and per-code
  (column) standard deviation, and save the running mean and standard
  deviation for visualization.
* Tune these six configurations on seed 0 using only val_knn:
  scalar (tau=0.08, K=1024), scalar (0.10, 1024), scalar (0.12, 1024),
  column (0.08, 1024), column (0.10, 1024), and scalar (0.10, 512).
* Select by validation accuracy, then report knn_eval for the selected method
  across seeds 0, 1, and 2; evaluate seed 0 only after selection.
* Save each run's weights, history, configuration, and results.csv row.

Constraints: no labels in the training loss, no more than 20 epochs per run,
and no changes to TinyViT or the fixed k-NN evaluation.
"""


def standardized_targets(logits, temperature):
    pass



def main():
    raise NotImplementedError(
        "Person 2 must implement the standardization experiment described in this file."
    )


if __name__ == "__main__":
    main()

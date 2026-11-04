"""TODO — Person 3: implement the mutual-information experiment.

Required work:
* Train without centering or temperature sharpening; use the same temperature
  for teacher and student.
* Implement cross-entropy plus ``-lambda_marg * H(P_bar)`` and
  ``+lambda_cond * mean(H(P_s))``. Compute ``P_bar`` from the concatenated
  student probabilities of both views (2B samples).
* Tune these six configurations on seed 0 using only val_knn:
  (lambda_marg=1.5, lambda_cond=0.1, tau=0.07, K=1024);
  (2.0, 0.2, 0.07, 1024); (2.5, 0.3, 0.07, 1024);
  (2.0, 0.2, 0.05, 1024); (2.0, 0.2, 0.10, 1024);
  (2.0, 0.2, 0.07, 512).
* Select by validation accuracy, then report knn_eval for the selected method
  across seeds 0, 1, and 2; evaluate seed 0 only after selection.
* Save each run's weights, history, configuration, and results.csv row.

Constraints: no labels in the training loss, no more than 20 epochs per run,
and no changes to TinyViT or the fixed k-NN evaluation.
"""


def mutual_info_loss(logits, temperature):
    pass



def main():
    raise NotImplementedError(
        "Person 3 must implement the mutual-information experiment described in this file."
    )


if __name__ == "__main__":
    main()

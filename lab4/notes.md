# DINO: centering and sharpening findings

The plots show the expected qualitative effects of removing centering and
sharpening. The four runs use 1,024 output codes. Entropy and KL are measured in
nats, so the maximum possible teacher entropy is `ln(1024) ≈ 6.931`.

## Final training measurements

These values come from the last logged training step, 1,940, in the saved
checkpoints. They describe the logged training batch, rather than the two test
images used in the bar plots.

| Variant | Teacher entropy | Teacher–student KL | Interpretation |
| --- | ---: | ---: | --- |
| Full DINO | 1.523 | 1.672 | Confident outputs with substantial variation between images |
| No centering | 0.149 | 0.177 | Strong concentration on very few codes; evidence of collapse |
| No sharpening | 6.926 | 0.0010 | Almost uniform outputs and close teacher–student agreement |
| No centering or sharpening | 6.740 | 0.0004 | Nearly constant, somewhat nonuniform outputs |

The no-centering run has not reached an exactly constant one-hot output:
its entropy and KL remain above zero. The overall trend is toward that form of
collapse. Without sharpening, the teacher entropy approaches its maximum,
indicating collapse toward uniform predictions. These trends agree with the
collapse study in the [DINO paper, Section 5.3](https://arxiv.org/html/2104.14294v2#S5.SS3).

## Why the distributions agree without sharpening

The student learns to match the teacher, while the teacher follows the student
through an exponential moving average of its weights. Centering suppresses
output codes that consistently dominate. Sharpening makes the teacher targets
more decisive and counteracts the tendency toward uniform predictions.
Removing sharpening leaves a route to agreement in which both networks produce
approximately the same uniform distribution for every image. The complementary
roles of centering and sharpening are described in the
[DINO paper](https://arxiv.org/html/2104.14294v2#S3.SS1).

For a uniform prediction, every code has probability approximately
`1/1024 ≈ 0.000977`. If both networks predict this distribution, their KL is
approximately zero. Agreement is easy to achieve this way, but the output
contains little information about which image was supplied.

The saved history supports this interpretation: without sharpening, the final
entropy of the batch-average teacher distribution is `6.9314`, while the mean
per-image entropy is `6.9256`. Their small difference indicates little variation
between images in that logged batch. With full DINO, these values are `5.6524`
and `1.5226`, respectively, indicating much greater variation between images.

## Does shared initialization explain it?

Shared initialization explains some initial similarity. Every variant starts
with the teacher copied from the student, so it cannot explain the different
outcomes across variants by itself.

Initially, identical networks produce identical distributions when given the
same image, equal temperatures, and a zero center. During training, however,
teacher and student receive different augmented views, the student changes
through gradient updates, and the teacher follows through its moving average.
The eventual near-zero KL without sharpening reflects the convergence toward
nearly image-independent outputs.

## Reading the figures

- Purple and teal bars represent **two test images evaluated by the teacher**.
  They are not teacher and student distributions. Teacher–student agreement is
  measured by the KL training curve.
- The bar plots display the 12 most active codes for each variant. Probabilities
  are computed over all 1,024 codes before selecting bars, and entropy and total
  variation distance also use all codes.
- The no-sharpening probability panels look almost empty because probabilities
  near `0.001` are displayed on a vertical axis ranging from 0 to 1.
- A higher KL in full DINO is plausible: the teacher uses a lower temperature
  than the student, and the logged comparison uses different augmented views.
  A rising KL alone does not establish that training is failing.
- Low KL alone does not establish collapse either. Here, the entropy and
  between-image variation provide additional evidence. Representation quality
  should be assessed using the prescribed k-NN evaluation.

## Generated figures

- [Teacher softmax stages for all four variants](teacher-softmax-stages.png)
- [Teacher entropy and teacher–student KL during training](entropy-and-teacher-student-KL.png)

## Person 2: target-side batch standardization

The six configurations were trained for 20 epochs each with seed 0. The
selected configuration was chosen using `val_knn` only: column-wise
standardization, `tau = 0.08`, `K = 1024`.

| Rank | Standard deviation | Temperature | K | Final `val_knn` |
| ---: | --- | ---: | ---: | ---: |
| 1 | Column-wise | 0.08 | 1024 | 0.4780 |
| 2 | Column-wise | 0.10 | 1024 | 0.4736 |
| 3 | Scalar | 0.08 | 1024 | 0.4560 |
| 4 | Scalar | 0.10 | 1024 | 0.4484 |
| 5 | Scalar | 0.10 | 512 | 0.4410 |
| 6 | Scalar | 0.12 | 1024 | 0.4356 |

### Selected configuration across seeds

The selected seed-0 tuning run was evaluated on the test set only after
selection. Seeds 1 and 2 were then trained with the same configuration.

| Seed | `val_knn` | `knn_eval` |
| ---: | ---: | ---: |
| 0 | 0.4780 | 0.4976 |
| 1 | 0.4888 | 0.4996 |
| 2 | 0.5020 | 0.5002 |

Across the three test runs, mean accuracy is **49.91%** with sample standard
deviation **0.14 percentage points**. The individual returned accuracies are
49.76%, 49.96%, and 50.02%. Relative to the task's reference figures, this is
about 1.11 points above standard DINO (48.8%) and 4.21 points above raw pixels
(45.7%).

### Collapse diagnostics and interpretation

For the selected configuration, the final logged training-batch values were:

| Seed | Teacher sample entropy (nats) | Batch entropy (nats) | Teacher–student KL (nats) |
| ---: | ---: | ---: | ---: |
| 0 | 0.53 | 5.94 | 2.863 |
| 1 | 0.55 | 5.94 | 2.894 |
| 2 | 0.55 | 5.99 | 2.897 |

The low per-image entropy indicates sharp teacher targets, while batch entropy
near 6 nats indicates substantially broader code use than in the scalar
standardization runs (whose final batch entropy ranged from 3.35 to 3.84).
For the selected runs, `exp(batch_entropy)` corresponds to roughly 380–400
effective codes in the logged batch, out of 1,024. These are batch-level
diagnostics, not measurements over the entire dataset.

The method standardizes teacher logits over both augmented views: it subtracts
each code's batch mean and divides by either a scalar or per-code batch
standard deviation. Teacher and student use the same temperature, and no
running DINO center is applied. The results are consistent with centering the
batch activity while preserving logit variation so the teacher targets remain
sharp.

This is empirical evidence, not a proof that every collapsed state is
impossible. If logits are exactly constant across a batch, their centered
variance is zero and the standardized targets become uniform. In these runs,
the entropy diagnostics and k-NN results show that this degenerate outcome was
not reached.

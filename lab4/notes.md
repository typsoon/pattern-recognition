# DINO: centering and sharpening findings

The plots show the expected qualitative effects of removing centering and
sharpening. The four runs use 1,024 output codes. Entropy and KL are measured in
nats, so the maximum possible teacher entropy is `ln(1024) ≈ 6.931`.

## Final training measurements

Reporting precision: the runner notebook prints epoch-20 entropies to two
decimals and KL to three. Use the rounded values below; do not imply additional
precision. The training loop processes 97 complete batches per epoch, so the
final epoch log is step 1,940. These metrics describe its final logged training
batch, not the two test images used in the bar plots.

| Variant | Teacher entropy | Batch entropy | Teacher–student KL | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Full DINO | 1.51 | 5.65 | 1.767 | Confident outputs with substantial variation between images |
| No centering | 0.20 | 0.89 | 0.247 | Strong concentration on very few codes; evidence of collapse |
| No sharpening | 6.93 | 6.93 | 0.001 | Almost uniform outputs and close teacher–student agreement |
| No centering or sharpening | 6.74 | 6.74 | 0.000 | Nearly constant, somewhat nonuniform outputs |

The no-centering run has not reached an exactly constant one-hot output: its
teacher entropy, batch entropy, and KL remain above zero. The low batch entropy
shows that code use is concentrated, but it does not establish that every
image uses the same code. Without sharpening, both entropies approach their
maximum, indicating collapse toward uniform predictions. These trends agree
with the collapse study in the
[DINO paper, Section 5.3](https://arxiv.org/html/2104.14294v2#S5.SS3).

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

At the runner's printed precision, the no-sharpening run ends with per-image
and batch-average entropies of `6.93` and `6.93`; full DINO ends at `1.51` and
`5.65`, respectively. The notebook output does not report additional decimal
places for these epoch summaries, so the values above should not be presented
as more precise than shown.

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

- [Teacher softmax stages for all four variants](plots/teacher-softmax-stages.png)
- [Teacher entropy and teacher–student KL during training](plots/entropy-and-teacher-student-KL.png)

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

## Person 3: mutual-information regularization

### Mathematical formulation and motivation

Without centering and sharpening (teacher and student share one temperature
`tau_t = tau_s = tau`), unconstrained self-distillation collapses into one of
two degenerate equilibria where code assignments become independent of the
input image:

1. **Single-code dominance (Collapse A):** every image maps to the exact same
   prototype code $\implies H(\bar{P}_s) = 0$ and $\text{mean}(H(P_{s,i})) = 0$.
2. **Uniform code dispersion (Collapse B):** every image yields a flat, uniform
   probability distribution across all $K$ prototypes $\implies H(\bar{P}_s) = \ln K$
   and $\text{mean}(H(P_{s,i})) = \ln K$.

Here, $P_{s,i}$ denotes the student distribution for image view $i \in \{1, \dots, 2B\}$,
and $\bar{P}_s = \frac{1}{2B} \sum_{i=1}^{2B} P_{s,i}$ is the empirical marginal
code distribution across the batch of $2B$ views. The method adds two explicit
entropy penalties to the DINO cross-entropy loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}} - \lambda_{\text{marg}} H(\bar{P}_s) + \lambda_{\text{cond}} \frac{1}{2B} \sum_{i=1}^{2B} H(P_{s,i})$$

When $\lambda_{\text{marg}} = \lambda_{\text{cond}} = \lambda$, this penalty is
strictly $-\lambda I(X; Z)$, the negative mutual information between the input
view $X$ and discrete code assignment $Z$. Maximizing the marginal entropy
$H(\bar{P}_s)$ penalizes code concentration across the batch (blocking Collapse A),
while minimizing conditional entropy $\text{mean}(H(P_{s,i}))$ forces confident,
sharp predictions for each individual sample (blocking Collapse B).

### Hyperparameter tuning on seed 0

Eight candidate configurations were trained for 20 epochs on seed 0. Configuration
selection relied exclusively on validation accuracy (`val_knn`); test evaluation
was performed only after tuning.

| Rank | $\lambda_{\text{marg}}$ | $\lambda_{\text{cond}}$ | Temperature ($\tau$) | $K$ | Final `val_knn` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.50 | 0.50 | 0.05 | 1024 | 0.4566 |
| 2 | 0.25 | 0.25 | 0.05 | 1024 | 0.4358 |
| 3 | 2.50 | 0.30 | 0.07 | 1024 | 0.3874 |
| 4 | 2.00 | 0.20 | 0.05 | 1024 | 0.3804 |
| 5 | 1.50 | 0.10 | 0.07 | 1024 | 0.3790 |
| 6 | 2.00 | 0.20 | 0.07 | 512 | 0.3788 |
| 7 | 2.00 | 0.20 | 0.10 | 1024 | 0.3784 |
| 8 | 2.00 | 0.20 | 0.07 | 1024 | 0.3774 |

Equal mutual-information weighting ($\lambda_{\text{marg}} = \lambda_{\text{cond}} = 0.50$)
with a lower temperature ($\tau = 0.05$) markedly outperformed the asymmetric
penalty schedules ($\lambda_{\text{marg}} \gg \lambda_{\text{cond}}$), which stagnated
near `val_knn ≈ 0.38`.

### Selected configuration across seeds

The selected configuration (`lambda_marg = 0.5, lambda_cond = 0.5, tau = 0.05, K = 1024`)
was then trained on seeds 1 and 2 and evaluated on the test set (`knn_eval`):

| Seed | `val_knn` | `knn_eval` |
| ---: | ---: | ---: |
| 0 | 0.4566 | 0.4669 |
| 1 | 0.4762 | 0.4851 |
| 2 | 0.4664 | 0.4749 |

Across the three runs, mean test accuracy is **47.56%** with sample standard
deviation **0.91 percentage points**. The individual test accuracies are
46.69%, 48.51%, and 47.49%. While outperforming untrained ViT (32.0%) and raw
pixels (45.68%), mutual-information regularization lags behind standard DINO
(48.76% notebook / 48.8% reference) and Person 2 target standardization
(49.91% ± 0.14%).

### Collapse diagnostics and interpretation

For the selected configuration, the final logged training-batch values were:

| Seed | Teacher sample entropy (nats) | Batch entropy (nats) | Teacher–student KL (nats) |
| ---: | ---: | ---: | ---: |
| 0 | 2.52 | 5.09 | 1.114 |
| 1 | 2.65 | 5.14 | 1.168 |
| 2 | 2.67 | 5.17 | 1.102 |

The diagnostics confirm that both collapse regimes were prevented:
- **Batch entropy** stabilized around 5.1–5.2 nats. While below the theoretical
  maximum ($\ln 1024 \approx 6.93$ nats) and lower than standardization (~5.95 nats),
  $\exp(5.15) \approx 172$ codes were actively utilized per batch, successfully
  precluding single-code dominance.
- **Per-sample entropy** decreased from ~6.0 nats at epoch 1 to ~2.5–2.7 nats at
  epoch 20, demonstrating that student distributions became moderately confident
  rather than flattening into uniform probabilities.
- **Teacher–student KL** settled around ~1.1 nats. Because teacher and student share
  the same temperature ($\tau = 0.05$) without centering, this divergence reflects
  genuine predictive disagreement across different augmented views.

### Why mutual information underperformed target standardization

Two structural factors explain why mutual information achieved lower representation
quality:

1. **Finite-sample marginal entropy bias:** The batch marginal $\bar{P}_s$ is computed
   over $2B = 1024$ views across $K = 1024$ prototypes. With an average occupancy of
   only 1 sample per bin, plug-in empirical entropy estimation suffers from substantial
   negative bias and high gradient variance.
2. **Gradient interference:** The mutual-information loss applies direct analytical
   gradients to the student's softmax logits, competing with the cross-entropy
   alignment to the EMA teacher. In contrast, target standardization operates
   exclusively on the teacher's targets *before* computing the distillation loss,
   leaving the student's learning dynamics harmonious and gradient-consistent.


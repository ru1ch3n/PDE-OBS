# Shared training template and archived attempt records

The benchmark uses a common data/task/metric interface. A common interface is not
a claim that all archived models used identical optimization histories.

## Shared template

- One model per PDE, method and training observation pattern: 7 x 7 x 9 settings.
- One deterministic split per PDE: 1,800 training identities (600 per regime),
  200 test identities (67/67/66), seed 20260804, no validation identities.
- The same nine identity-seeded masks across methods. No mask resampling each epoch.
- Stationary reconstruction receives the masked solution, mask and geometry.
  Dynamic forecasting observes stored frame 0 and predicts frames 1, 2 and 3
  autoregressively, with no future ground truth fed into the model.
- The supervised objective is the minibatch mean of per-record relative-L2 over
  the complete target; temporal frames are scored jointly. PINO additionally uses
  its recorded physical residual. The final recorded checkpoint is retained,
  rather than a checkpoint selected by held-out test error.
- The framework supports fixed epoch budgets, full-state continuations and
  training-loss early stopping where the recorded protocol enables it. The
  registry, resolved configuration and completion record identify the recipe.

## What actually differs in this archive

The immutable source of per-setting facts is
[`results/archived_campaign/index.json`](../results/archived_campaign/index.json).
Its `records` object is keyed by `pde/method/train_view`. Preserve `checkpoint_id`,
`training_cohort`, `planned_epochs`, `actual_epochs`, `stop_reason`, protocol
evidence, configuration identity and recorded health information together.

| Recorded cohort | Settings | Interpretation |
|---|---:|---|
| `original500` | 257 | Original 500-epoch protocol |
| `original500_recovered` | 66 | Original protocol with recorded continuation |
| `fixed200_recovered` | 36 | Fixed 200-epoch protocol with continuation |
| `budget200_min120` | 44 | At most 200; loss-based plateau counting starts after epoch 120 |
| `budget200_patience10_no_floor` | 10 | At most 200; earlier patience protocol without the minimum-120 floor |
| `interrupted_or_recovered` | 28 | Retained interrupted, salvaged or exceptionally terminated checkpoints |
| Total | 441 | Distinct evaluated model identities, not submission attempts |

For `budget200_min120`, a strict new historical minimum of finite training data
loss resets the post-120 counter. Ten complete non-improving epochs from epoch
121 onward stop training; the earliest plateau stop is epoch 130. No validation
or test loss selects this stop. The older no-floor protocol is not relabelled as
the newer one. Ordinary scheduler interruption and checkpoint recovery are not
plateau stopping. An interrupted/salvaged checkpoint is not described as having
completed its planned budget.

Bounded training-parameter amendments and training-loss admission decisions also
occurred. Their original configuration/protocol evidence remains in each record.
Do not describe every early endpoint as loss-plateau stopping, or a repaired
attempt as an unchanged rerun. Epoch ceilings are 500 for 350 settings and 200
for 91, but actual epochs must be read independently. There are 324 checkpoints
with 500 actual epochs, and 23 PDE-method pairs for which all nine views reached
500. The latter are the manuscript's all-500 sensitivity subset. That subset
does not equalize optimizer histories, capacity, compute or training information.

These are retrospective results from retained attempts, not a preregistered,
uniform-budget architecture leaderboard. A complete grid does not remove
historical selection. Finite high errors, warning history and cross-view
instability must remain visible. One seed cannot establish seed variability.

## Verification does not change training

The separate [prediction verification](prediction_verification.md) reuses the
exact evaluated weights and fixed test identities. It performs no optimizer
updates and does not choose a replacement checkpoint or tune hyperparameters.
Its outputs cannot erase the original cohort or turn a salvaged checkpoint into
standard budget-complete training.

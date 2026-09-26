# Tasks and information permissions

The complete record is stored for supervision and evaluation. The model must
receive only the declared visible input. These are separate concepts: a target
being present in a batch is not authorization to pass it into the model.

## Task contract

Let `c` be the complete PDE condition, `u_t` a complete state, `M` the fixed
observed-site mask, and `g` geometry. Multiplication by `M` below includes a
separate mask channel rather than treating zero-filled values as fully known.

| Task | Predictor input | Supervised target | Status in the paper |
|---|---|---|---|
| Recovery | `M*u_t`, `M`, `g` | Complete `u_t` | Main static experiment |
| Forward | `M*c`, `M`, `g` | Complete selected solution `u_t` | Implemented extension |
| Inverse | `M*u_t`, `M`, `g` | Complete condition `c` | Implemented extension |
| Rollout | Masked observed history, its mask, `g` | Complete future states | Main dynamic experiment |

`BenchmarkDataset` uses HWC for a single state and THWC for a trajectory; batches
add a leading axis. The neural preparation path converts layouts as required by
the model. Targets keep full spatial coverage for the declared supervised
objective. The primary NS state is vorticity, and an explicit state-conversion
setting is a different representation.

## Three distinct information paths

1. **Predictor-visible:** observations, the mask, geometry, and any explicitly
   allowed metadata/context of the selected adapter.
2. **Training supervision:** complete targets used by the loss. Backpropagation
   from a complete target is not itself test-time leakage.
3. **Privileged training-loss context:** a declared physics-informed objective
   may use complete PDE coefficients/sources for its residual. PINO's
   `pde_condition` is used in this path, not as an ordinary recovery input.

The dataset may also retain an `initial_state` for classical reduced-order
fitting or audit context. The learned rollout forward path does not use this
complete initial state as a hidden replacement for masked input.

PINO is therefore not identical in training information to a purely data-loss
model. Its residual definition, weights, boundary assumptions, and full-precision
requirement must be reported. Test-time predictions must still obey the declared
observation contract.

## Paper rollout

The paper uses one observed state and predicts the next three stored frames:

```text
visible input: (M*u0, M, geometry)
u1_hat = model(visible input)
u2_hat = model(u1_hat)
u3_hat = model(u2_hat)
loss/score compares predictions to u1, u2, u3
```

Teacher forcing is zero. Future ground truth is not fed back. The built-in
autoregressive wrapper does not detach each predicted state by default, so
training gradients can propagate through the recurrence. Subsequent predicted
states are fed forward as predictions, not mixed with unobserved future truth.
Do not substitute the generic dataset horizon of eight or a one-step experiment
for this history-1/horizon-3 protocol.

For a dataset that already slices its target to the future window, evaluation's
target offset is zero. The associated time indices in the strict contract remain
the physical record's selected target-frame indices. Joint error over the three
future frames and mean per-horizon error are distinct metrics.

## Built-in checks versus arbitrary plugins

The release tests exercise the built-in data/predictor boundary with fixed masks:
changing hidden values or a future target while leaving permitted observations
unchanged must not change ordinary predictions. Consult the release receipt for
which model/path was executed. This is a targeted test, not a proof about all
possible third-party plugins.

A plugin can receive keyword arguments or metadata, so it must declare and audit
its own information access. Registration, import success, or a valid output shape
does not certify permission compliance. Do not pass the target, full condition,
complete initial state, or privileged loss context into an ordinary predictor to
improve a score.

## Training and model selection

The paper partition has 1,800 training and 200 test identities with no validation
set. Complete training targets support the loss; held-out test identities are
not used for checkpoint selection, schedule selection, or early stopping.
The final/last checkpoint is the relevant artifact for each declared attempt.

The generic trainer can support other splits and budgets. Its `val_loss` field
can equal training loss when no validation loader exists; the field name is not
evidence that a validation set was used. Historical 500-epoch, fixed-200,
200/min120, and interrupted/recovered attempts are separate protocol cohorts.
See [Reproducing the paper](reproducing_the_paper.md).

## Optional projection

Static recovery can optionally replace predicted observed entries with the
available observation **after raw prediction checks**. This is explicitly
reported data-consistency projection, not a reason to hide non-finite raw values.
It applies only where `M` states values were observed. It must not copy future
ground truth into a rollout or disclose missing entries to a model.

The strict scorer reports raw and projected status/metrics separately where the
projection option is selected. Hidden-only normalization is separately named;
when there are no hidden entries it is `not_applicable`, not an artificial zero.

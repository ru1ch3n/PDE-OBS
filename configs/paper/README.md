# Paper configurations

These files separate the paper's selected scope from generic demos and retained
historical configurations:

- `protocol.yaml`: seven PDEs, seven public learned methods, exact 1800/200
  split with no validation, static recovery and history-1/horizon-3 rollout.
- `observations.yaml`: the nine 128 x 128 masks and exact realized counts.
- `training_cohorts.yaml`: original500, fixed200, budget200/min120 and
  interrupted/recovered semantics. This file is metadata, not a stopping plugin.

They intentionally use descriptive schemas. Do not give them to `pdeobs train`
as if they were complete per-method runnable configurations, or to the frozen
legacy `one_setting.load_campaign` validator. The original campaign and method
registry are preserved separately so that their historical values are not
silently rewritten. The original resolver only accepts three diagnostic or 500
production epochs; later budget adapters are different versions.

Paths in the public description are relative defaults or caller-supplied process
environment variables. There is no required author filesystem or cluster.
See [the reproduction guide](../../docs/reproducing_the_paper.md) for exact
identity selection, row materialization, method parameters, and score provenance.

# Optional extension scope

The examples and interfaces here are broader than the paper's primary learned
recovery/rollout comparison. Implemented is not the same as paper-evaluated.

`forward_poisson_fno.yaml` and `inverse_darcy_unet.yaml` are portable wrappers
around the preserved generic experiment definitions. They retain those original
scientific parameters (including their generic observation/split settings),
with explicit caller-controlled root defaults. They are not the paper's
1800/200 nine-view protocol and are not tiny smoke tests. Inspect the resolved
configuration and provide matching data before training. Neural examples need
the optional training dependency.

`interfaces.yaml` names implemented optional Python interfaces and the status of
their CLI paths. Retrieval, routing, upstream wrappers and broader task support
are not counted as extra primary algorithms or as completed experiments.

Use the installed E6 demo for the low-cost, tested custom-mask/model route;
see [Extending PDE-OBS](../../docs/extending.md).

# Public release - 26 September 2026

The canonical public repository is https://github.com/ru1ch3n/PDE-OBS.
The project homepage remains https://ru1ch3n.github.io/PartialObs--PDEBench/.
The separate anonymous review copy is not redirected or altered by this release.

The public copy carries the existing v0.2.1 implementation, configurations, tests
and recorded results. No numerical kernel, model, training protocol, mask, split,
score, checkpoint or result tensor has been changed for publication. The Git
history begins with this release snapshot rather than publishing private development
history. Historical review-overlay notes and acceptance records are preserved as
historical evidence, not as claims that those tests were rerun for publication.

Current scientific evidence:

- `results/prediction_verification_20260924/`: 441 retained models, 3,969 complete
  200-record evaluation blocks and all per-record errors. Finite adverse scores
  remain in the results. This is not a uniform-budget architecture leaderboard.
- `results/revision_v2/`: five-model strict mixed-pattern comparisons, paired
  test-identity bootstrap intervals, training inventory and 1,701 stationary
  observed/hidden and projection diagnostic blocks.
- `results/numerical_quality/`: production-discrete residual checks, not an
  independent reference-solution error or grid-convergence study.
- `results/archived_campaign/`: historical scalar archive, not the source of
  current paper scores.

Data and weights remain in their separately licensed Hugging Face deposits;
large tensors, checkpoints, credentials and local working directories are not
added to this repository. External availability and reproducibility are separate
from prediction validity and scientific quality.
Current public locations are the [dataset](https://huggingface.co/datasets/ru1ch3n/PDE-OBS)
and [441-checkpoint model repository](https://huggingface.co/ru1ch3n/PDE-OBS).
The dated receipts under `results/` retain the original deposit identifiers as
historical evidence; they are not rewritten to imply a second complete file audit.

## Paper

`paper/PDE_OBS_preprint.pdf` is the named public version. Only public-release
metadata, author affiliations, corresponding-author marks, repository wording
and preprint layout change relative to the anonymous manuscript. Scientific text,
results, figures, tables and references are preserved. The public PDF uses the
single-column `arxiv-style` template; its MIT-licensed style and license file are
in `paper/PDE_OBS_arxiv_source.zip`. The original anonymous manuscript is retained
separately. A public preprint is not an accepted ICLR paper.

An arXiv identifier has not yet been assigned. Do not fabricate one in citations.
The root MIT license applies to the software; it does not choose an arXiv license
for the manuscript or override data, checkpoint or third-party asset terms.

## Install and inspect without starting an experiment

```bash
git clone https://github.com/ru1ch3n/PDE-OBS.git
cd PDE-OBS
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install .
pdeobs --help
```

These installation commands do not download the corpus or start paper-scale
training. See `docs/installation.md` and `docs/reproducing_the_paper.md` for
explicit, separately budgeted workflows.

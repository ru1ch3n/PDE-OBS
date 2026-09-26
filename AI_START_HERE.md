**Start with this file, not every Markdown file.** This is PDE-OBS v0.2.0 with
separately documented review overlays and a completed prediction-level result cut.
Current scores, per-record errors and audit bindings are in
`results/prediction_verification_20260924/` (441 checkpoints, 3,969 full-200 valid
blocks). The original scalar archive remains unchanged and is not a current-score
fallback. No production tensors or weights are embedded in this tree. Validity,
score agreement, predictive quality and external anonymous access are separate.
See `AI_PATCH_NOTES.md` and `docs/prediction_verification.md`.

| Intent | Read next | Execute only after checking budget |
|---|---|---|
| Copy-paste commands by cost level | `docs/ai/QUICKSTART.md` | levels 1-3 only; level 4 needs authorization |
| Understand the benchmark | `docs/ai/PROTOCOLS.md` | `python tools/ai_assist.py doctor --json` |
| Find the implementation | `docs/ai/repository_index.json` | `python tools/ai_assist.py context --task overview` |
| Run a tiny example | `docs/ai/WORKFLOWS.md` | `python tools/ai_assist.py smoke --out ../pdeobs-ai-smoke-001` |
| Add/inspect a model | `src/pdeobs/api/specs.py`, `docs/extending.md` | `python -m pdeobs easy describe fno --json` |
| Score observations/predictions | `src/pdeobs/strict_score.py`, `src/pdeobs/api/evaluate.py` | use explicit targets/IDs |
| Reproduce an actual paper row | `configs/paper/README.md`, `docs/reproducing_the_paper.md` | `paper-row` is expensive; never auto-launch |
| Check existing paper materials | `docs/paper_evidence.md` | `python -m pdeobs.evidence check --bundle <manifest>` (read-only) |
| Recompute current results without inference | `results/prediction_verification_20260924/README.md`, `tools/prediction_analysis/` | CPU metadata arithmetic only; do not relaunch verification to reproduce a table |
| Read or regenerate the archived results index | `results/archived_campaign/README.md`, `docs/archived_results.md` | `python tools/archived_results.py scan` on the archive host (read-only), then `pair` and `index` into a new directory |
| Which acceptance record applies | `docs/ai/ACCEPTANCE_INDEX.md` | none; the data is `docs/ai/acceptance_index.json` |
| Assess available evidence | `docs/ai/evidence_status.json`, `docs/ai/AUDIT.md`, `docs/ai/ACCEPTANCE_REPORT_overlay2.md` | do not infer status from a filename |

**Boundaries in one line:** `doctor`, `context` and `evidence check` are read-only; `smoke` and
`evidence export` write only into a new directory you name; nothing here trains at paper scale,
downloads, or launches jobs. A software demo row (`cohort: demo`) or a smoke receipt is never paper
evidence, and the checker refuses to export it as such.

No need to install an agent server, external account, or MCP service. These are local, inspectable
files/commands usable by any assistant with repository access. Different assistants may not
read `AGENTS.md` automatically; explicitly point them here.

# Prediction-level verification of archived checkpoints

Historical scalar metrics remain in `results/archived_campaign/`. No historical
prediction tensors were retained, so those original arrays cannot be checked
retrospectively. The inference-only adapter in
`src/pdeobs/archived_verification.py` performs a **new, separately identified
verification** using the same evaluated weights, data identities and masks.
It does not overwrite the archive or claim that old predictions were retained.

## Completed result version (2026-09-25)

The inference-only run and terminal audit are complete for 441 models and
3,969 blocks, each with all 200 identities and finite predictions/targets.
Version `pdeobs-prediction-verification/20260925-v1` is in
[`results/prediction_verification_20260924/`](../results/prediction_verification_20260924/README.md).
It retains all 793,800 per-record cell errors, dynamic horizon errors, input
hash bindings and new statistics. Zero blocks are invalid and zero models
failed this verification; this is not a scientific quality PASS.

The terminal audit freshly rehashed all 3,969 prediction HDF5 files, authenticated
the strict scorer's full-array checks against those digests, regenerated masks,
and independently checked identities, geometry and physical-time mappings.
It rehashed all 84 selected data shards and model/config inputs. No forward pass
or training was repeated. The fixed 13-pair/117-model original500 configuration
subset was rechecked unchanged. Raw tensors remain outside Git.

3,646 blocks meet the original comparison rule, another 314 differ by at most 1%,
and nine by more than 1%. The 1% reporting convention was adopted after initial
results; it is not preregistered or a validity gate. Old strict fields remain
unchanged. All valid new scores are used regardless of direction; old scores
remain only in audit comparison and archives. No closer-agreement replay is
required. Specific numerical causes of differences are unconfirmed.

Use the CPU aggregation commands in the result README to reproduce current
tables. The inference command below is an optional expensive workflow, not a
required step for these already-complete statistics.

## Contract

1. Bind the original checkpoint digest in the archive to the stripped release
   checkpoint digest through the release manifest. Rehash the actual released
   checkpoint and resolved configuration. The two checkpoint byte hashes are
   not interchangeable.
2. Rehash the selected data shards and original dataset summary. Reconstruct the
   deterministic 1,800/200 split, and compare all scientific split fields.
3. Evaluate the fixed model under all nine original views on all 200 held-out
   identities. Preserve the original inference batch size, task, seed and mask
   definitions. Dynamic inference feeds its own predictions forward.
4. Retain predictions, targets, observations, masks, geometry, sample identities
   and time metadata in a new HDF5 file per block. Never omit an adverse example.
5. Score those files independently with the unchanged `pdeobs-strict-v1` scorer.
   Reject missing, repeated, misaligned or nonfinite predictions. Large finite
   errors remain valid poor outcomes. Preserve invalid arrays and explicit errors.
6. Retain every block's contract, score, checkpoint/data binding, output hashes,
   runtime and comparison with its archived scalar. Do not select models or
   hyperparameters based on the comparison.

Stored-frame **positions** are `[0]` for recovery and `[1,2,3]` for forecasting.
Different physical regimes can use different internal solver increments, so
the corresponding solver-frame indices need not be identical across records.
The adapter binds each identity's solver indices and physical times separately;
tests check both mappings and fail on altered time metadata.

The comparison threshold is fixed at `atol=1e-7, rtol=1e-4` before collection.
It labels scalar reproducibility, **not scientific quality**. A finite block can
pass array checks while disagreeing with its archived score. Cross-hardware or
software-version differences are possible explanations to investigate, not
automatic excuses to widen the tolerance or call all differences harmless.

## One-model command

Use a new output directory and a GPU that you are authorized to use. This command
does not request scheduler resources or establish a cross-process execution
lock; a multi-worker dispatcher must enforce one execution per model identity.
Use `--device cpu` for a CPU run, or an assigned `cuda:0` with PyTorch installed.

```bash
python tools/verify_archived_predictions.py \
  --identity poisson/fno/random_50pct \
  --index results/archived_campaign/index.json \
  --models-manifest /path/to/models_manifest.cluster-A.json \
  --model-root /path/to/the/model/directory \
  --data-root /path/to/the/original/data \
  --data-manifest /path/to/release_manifest.json \
  --campaign configs/campaign/all_pde_one_setting_10method_9x9.yaml \
  --output /path/to/new-verification/poisson-fno-r50 \
  --device cuda:0
```

The model directory contains `resolved.yaml` and `checkpoints/last.pt`.
The published corpus is a de-identified re-emission of the generated one: shard attributes and
sidecars have placeholders in place of the generation provenance, numerical arrays are
bit-identical. The public manifests list the published digests only; `results/public_deposits/release_map.json`
in this repository binds the generation-time digests recorded in the archived index and in
`dataset_bindings.json` to the published ones. The tool reads that map for `summary.json` and for the
checkpoint identity (`--release-map`, default: the file in this repository); the shard digests come from
the release manifest, which already lists the published files.
The data root contains the original `summary.json` and family/boundary/condition
shards. A weights-only release is sufficient for inference, not full-state
training continuation. No token, remote login, scheduler command or automatic
download is embedded in this tool.

```bash
python -m pytest -q tests/test_archived_verification.py tests/test_strict_inference.py
```

These synthetic tests check the adapter, not all production models. A launched
441-model plan is not a completed audit. Report inference completion, full-200
array validity, scalar agreement and predictive quality separately; preserve
partial work and failures with their original denominators.

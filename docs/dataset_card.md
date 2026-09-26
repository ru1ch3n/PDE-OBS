# Dataset card

What the records are, how to obtain them, and what this release does not distribute.

This card answers four questions that [Physical records, identities, and storage](data_schema.md)
deliberately does not: what is in the corpus, how large the design is, how a
reader obtains it, and under what terms. `data_schema.md` remains the normative
description of the on-disk layout; the numerical routes are described in
[Numerical systems and generation](numerical_solvers.md) and are not repeated here.

## The factorial design in one table

| Axis | Values | Count |
|---|---|---|
| PDE family | `darcy`, `poisson`, `helmholtz`, `heat`, `reaction_diffusion`, `burgers`, `navier_stokes` | 7 |
| Boundary protocol | `dirichlet`, `neumann`, `periodic`, `robin_obstacle` | 4 |
| Condition setting | ten named field constructions (`S0`-`S9`) | 10 |
| Records per macrodomain | 2,000, **divided across** the three parameter regimes | 2,000 |

A *macrodomain* is one (family, boundary, setting) cell. The full design is
therefore 7 x 4 x 10 x 2,000 = **560,000 planned physical records**. The 2,000
records of a macrodomain are divided across the three regimes (`low`, `medium`,
`high`) and are **not** multiplied by three: the full-tier regime allocation is
667 / 667 / 666 (`configs/dataset/default.yaml`, `splits.regime_allocation_full`;
the same numbers follow from `allocate_counts` in `../src/pdeobs/splits.py`).

The selected paper slice fixes one boundary and one setting per family, so it
uses seven macrodomains: 7 x 2,000 = **14,000 underlying physical records**.
Nine observation views over those records are not nine new physical solutions.

These are planned counts. This release contains no corpus and reports no
execution of the full design: `release/validation_status.json` records
`"paper_scale_original500_training": "not_executed"`. The corpus itself is now
published and hash-verified (`"public_upload_verified": true`, see
[Public release](public_release.md)); what this tree does not contain is the
corpus, not a claim that none exists.

### How the design maps onto shards

The design has 280 macrodomains, hence 280 x 3 = **840 parameter-regime nodes**.
The number of shard files is a function of `shard_size` alone, so two shard
counts appear in the tree and both are correct for their own configuration:

| Dataset configuration | `shard_size` | Shards per regime node | Full-design shards |
|---|---|---|---|
| `configs/dataset/default.yaml` | 700 | 1 (667 <= 700) | 840 |
| `configs/dataset/numerics_full_t15.yaml` | 200 | ceil(667 / 200) = 4 | 3,360 |

The paper-tier value is the live one: `expected_shards: 3360` is declared in the
campaign configurations under `configs/campaign/` and asserted in
`../tests/test_campaign.py`. `../tests/test_presets.py` loads
`configs/dataset/numerics_full_t15.yaml` and asserts that the resulting plan has
3,360 jobs, 560,000 samples in total, and no job larger than 200 samples.

## Per-family record table

Parameter values are unitless numbers on the code's unit domain; no SI units are
assigned anywhere in the repository. For the discretizations, solver identities
and their caveats, see [Numerical systems and generation](numerical_solvers.md).

| Family (factor ID) | Equation solved | What `condition[H,W,1]` means | Static / temporal | Generator-default stored frames | Regime parameter (low / medium / high) | State stored in `trajectory` |
|---|---|---|---|---|---|---|
| Darcy (F0) | `-div(a grad u) = f`, fixed deterministic zero-mean sine-mix forcing | **Coefficient** `a` (positive; the max/min ratio *targets* the requested contrast, see note below) | static | `T = 1` | coefficient contrast 2 / 8 / 32 | scalar `u` |
| Poisson (F1) | `-Laplace u = f` | **Source** `f` (regime amplitude x setting field; mean-projected for periodic and Neumann, by two different projections, see note below) | static | `T = 1` | source amplitude 0.5 / 1.0 / 2.0 | scalar `u` |
| Helmholtz (F2) | `-Laplace u - k^2 u = f` (real, nominal BVP) | **Source** `f` | static | `T = 1` | wavenumber `k` 0.5 / 1.0 / 2.0 | scalar `u` |
| Heat (F3) | `u_t = alpha Laplace u`, final time 0.25 | **Initial state** `u(t0)` | temporal | `T = 9` | diffusivity `alpha` 0.005 / 0.02 / 0.08 | scalar `u` |
| Reaction-diffusion (F4) | Allen-Cahn `u_t = 0.008 Laplace u + r(u - u^3)`, final time 0.8 | **Initial state** (0.8 x setting field) | temporal | `T = 9` | reaction rate `r` 0.5 / 1.5 / 4.0 | scalar `u` |
| Burgers (F5) | two-dimensional **scalar** `u_t + u(u_x + u_y) = nu Laplace u`, final time 0.35 | **Initial state** `u(t0)` | temporal | `T = 9` | viscosity `nu` 0.04 / 0.015 / 0.004 | scalar `u` |
| Navier-Stokes (F6) | vorticity-streamfunction form, final time 0.4 | **Initial vorticity** (regime scale x setting field) | temporal | `T = 9` | viscosity `nu` 0.02 / 0.008 / 0.0025 **and** initial vorticity scale 2 / 3 / 4 | scalar vorticity `omega`, never velocity or pressure; `trajectory` always has one state channel |

Notes that the tables above do not make explicit:

- The "Generator-default stored frames" column reports the low-level generator
  defaults `STATIC_TIME_STEPS = 1` and `TEMPORAL_TIME_STEPS = 9`
  (`../src/pdeobs/pdes/common.py`). They are **not** the dataset-level default:
  `configs/dataset/default.yaml` sets `stored_trajectory_steps: 15`, so a
  dataset built from the shipped configuration stores 15 temporal frames, not 9.
  Static families reject any request for `T != 1` (`resolve_time_steps`).
- The dataset-level default resolution is 128 x 128, and the paper-tier temporal
  profile stores 15 ordered frames selected by an integer stride from a denser
  in-memory trajectory (`configs/dataset/numerics_full_t15.yaml`), never by
  interpolation.
- **Darcy contrast.** The coefficient is built as
  `exp(0.5 * log(contrast) * bounded_latent)`, so the realized max/min ratio
  depends on the setting field and need not equal the regime value. Both numbers
  are stored as separate parameters: `requested_coefficient_contrast` (the
  regime value) and `realized_coefficient_contrast`, the latter computed from
  the float32-**stored** coefficient array (`../src/pdeobs/pdes/darcy.py`).
- **Poisson mean projection.** The two compatibility projections are not the
  same operation. For `periodic` the full-field mean is removed
  (`source -= mean(source)`); for `neumann` only the **interior** mean is removed
  (`source -= mean(source[1:-1, 1:-1])`), because the bounded discrete operator
  solves for interior nodes (`../src/pdeobs/pdes/poisson.py`).
- All ten settings are crossed with all seven families. The family alone decides
  whether the shared field is read as a coefficient, a source or an initial
  state; **no per-record metadata field records that physical role**, so the
  coefficient-vs-source distinction is implied by the family name.
- The ten settings are `smooth_grf` (S0), `medium_grf` (S1), `rough_grf` (S2),
  `low_frequency_fourier` (S3), `multi_frequency_fourier` (S4), `gaussian_blobs`
  (S5), `piecewise_blocks` (S6), `threshold_level_set` (S7), `dipole_vortex_pair`
  (S8), `front_ring_shock` (S9) (`../src/pdeobs/settings.py`). The three
  Gaussian-random-field settings differ only in correlation length
  (0.22 / 0.09 / 0.025).

## Generation method, family by family

Every record in this benchmark is produced by a solver in this repository. Nothing is
downloaded, scraped or converted from another corpus, and no surrogate is used in place of a
solve. This section states, for each family, the numerical route that produces the stored
field and the command that regenerates it.

### Invariants shared by all seven routes

- **Identity and seed.** A record's identity is derived from its case specification, not from
  its position in a file, so the same specification reproduces the same record on the same
  software stack. `--seed` (default `20260804`) seeds the condition-field draw; the split is a
  separate deterministic function of the identity
  ([`src/pdeobs/splits.py`](../src/pdeobs/splits.py)).
- **Elliptic tolerance.** The default generator routes solve to relative tolerance `1e-9`.
  Iteration bounds depend on family and grid. **Failure to converge raises**; a partial solve is
  never returned and never relabelled as a solution.
- **Time stepping is not the storage cadence.** Internal steps are chosen for stability; the
  Navier-Stokes periodic route additionally bounds its internal step by `1e-4`. Stored frames are
  selected from computed frames **by an integer stride, never by interpolation**, so one stored
  interval is generally many internal steps.
- **Storage.** `float32`, gzip level 4, written through the resumable shard protocol described in
  [Physical records, identities and storage](data_schema.md). The paper tier stores 15 ordered
  frames for temporal families and one frame for static families.
- **Quality gate.** Generation runs under a quality profile (`report`, `strict` or `publication`)
  that measures a normalized PDE residual per sample; `strict` and `publication` reject records
  that fail the calibrated check rather than storing them.

### The seven routes

| Family | Numerical route as implemented | Final time | Regime parameter |
|---|---|---|---|
| Darcy | Sparse variable-coefficient flux operator with arithmetic face coefficients, solved by diagonally preconditioned conjugate gradients | static | coefficient contrast 2 / 8 / 32 |
| Poisson | Sparse elliptic solve, with the compatibility (zero-mean) projection applied for periodic and Neumann cases; the two projections differ and are documented above | static | source scale 0.5 / 1 / 2 |
| Helmholtz | Real indefinite sparse solve using MINRES, because the reaction term makes the operator indefinite; damping is zero and a nonzero damping request is rejected | static | wavenumber `k` 0.5 / 1 / 2 |
| Heat | Fourier exponential diffusion for periodic data; Crank-Nicolson for bounded data | 0.25 | diffusivity 0.005 / 0.02 / 0.08 |
| Reaction-diffusion | Strang splitting: an analytic reaction half-step around a diffusion step (Allen-Cahn, not Gray-Scott) | 0.8 | reaction rate 0.5 / 1.5 / 4 |
| Burgers | Dealiased spectral convection for smooth periodic fields, Rusanov fluxes for nonsmooth or bounded cases, SSPRK2 nonlinear stages with diffusion; two-dimensional **scalar**, not a two-component velocity system | 0.35 | viscosity 0.04 / 0.015 / 0.004 |
| Navier-Stokes | Vorticity-streamfunction form: periodic dealiased vorticity stepping, or bounded / obstacle-aware streamfunction-vorticity routes. Records store **scalar vorticity**, never velocity or pressure | 0.4 | viscosity 0.02 / 0.008 / 0.0025 and initial vorticity scale 2 / 3 / 4 |

Route selection is a property of the boundary protocol and the field, not a user choice: a
periodic smooth field takes the spectral route, a bounded or nonsmooth one takes the
finite-volume route. [Numerical systems and generation](numerical_solvers.md) documents each
route's caveats and what the repository does **not** claim about them, including that no
grid-convergence study against an independent reference solution ships here.

### Regenerating the corpus

The full factorial is a two-step, manifest-driven flow:

```bash
pdeobs plan --tier full --output plan.jsonl          # 7 families x 4 boundaries x 10 settings x 3 regimes
pdeobs generate --plan plan.jsonl --output ./pdeobs-full --quality-profile strict
```

The paper slice is much smaller than the full factorial, because the protocol fixes one boundary
and one setting per family ([`configs/paper/protocol.yaml`](../configs/paper/protocol.yaml)):
`dirichlet` + `smooth_grf` for Darcy, Poisson and Helmholtz; `periodic` + `smooth_grf` for Heat,
Reaction-diffusion, Burgers and Navier-Stokes. One family, one regime, at paper resolution:

```bash
pdeobs generate-case --pde heat --boundary periodic --setting smooth_grf \
    --param-regime medium --num-samples 667 --resolution 128 --time-steps 15 \
    --seed 20260804 --root ./paper-data
```

Measured sizes of the frozen release, for planning disk and transfer:

| Corpus | Shards | Size on disk |
|---|---:|---:|
| Full factorial (840 family/boundary/setting/regime nodes, 4 shards each) | 3,360 | 235.1 GiB |
| Paper slice (7 nodes x 3 regimes, 4 shards each) | 84 | 6.4 GiB |

`--num-samples` on `generate-case` is the **per-regime** count, so the value above is one regime's
share of the macrodomain; `pdeobs plan --tier full` emits all 840 jobs with the allocation already
applied and is the authoritative route. The full tier stores 2,000 records per macrodomain,
allocated 667 / 667 / 666 across the three regimes and written at `shard_size: 200`
([`configs/dataset/numerics_full_t15.yaml`](../configs/dataset/numerics_full_t15.yaml)), which is
where the four-shards-per-node figure comes from.

## Boundary protocols and geometry

Four protocols are available to every family. The canonical internal name for
the fourth is `robin`; it is published at dataset level as `robin_obstacle`
(`BOUNDARIES` in `../src/pdeobs/generation.py`, with `BOUNDARY_ALIASES` in
`../src/pdeobs/pdes/common.py` mapping every accepted spelling onto the internal
name).

| Published label | Accepted aliases | Scalar families | Navier-Stokes | Geometry mask |
|---|---|---|---|---|
| `periodic` | `b2`, `wrap` | periodic operator; zero-mean compatibility where the operator has a nullspace | periodic vorticity, spectral streamfunction inversion | all-zero |
| `dirichlet` | `b0`, `no_slip`, `dirichlet_no_slip` | zero values at the outer stored grid layer | closed no-slip walls, Thom wall vorticity | outer wall ring |
| `neumann` | `b1`, `free_slip`, `zero_flux`, `neumann_free_slip` | zero-normal-difference relation, compatibility where necessary | free-slip walls (zero wall vorticity) | outer wall ring |
| `robin_obstacle` | `b3`, `robin`, `mixed`, `obstacle`, `mixed_obstacle`, `robin_mixed_obstacle` | mixed scalar relation: Dirichlet on the horizontal walls, homogeneous Robin on the vertical walls (`alpha = 1.0`, `beta = 0.15`) | closed-wall domain with a seeded circular obstacle, masked streamfunction/vorticity route | outer wall ring plus, **for Navier-Stokes only**, an interior disc |

What `robin_obstacle` does and does not assert:

- It **does** assert a mixed scalar relation for the elliptic and diffusive
  families, with the coefficients above.
- It **does not** assert that a scalar Robin condition is applied to vorticity,
  and it is not an inflow/outflow channel.
- **Every freshly generated `robin_obstacle` Navier-Stokes record nevertheless
  carries an unused `inflow_speed` parameter.** This is current behaviour, not
  historical residue: the generator takes `inflow_speed: float = 0.08`,
  validates it as non-negative, and writes
  `"inflow_speed": float(inflow_speed) if boundary == "robin" else 0.0` into
  `parameters` for every record, while none of its three integration branches
  (periodic, bounded, masked) reads the value
  (`../src/pdeobs/pdes/navier_stokes.py`). The code that would consume it lives
  in multi-channel velocity/pressure routines — a channel-flow helper in
  `../src/pdeobs/pdes/common.py` that carries its own, disagreeing default of
  `0.75` and is never imported by the Navier-Stokes generator, and replay
  branches in `../src/pdeobs/quality.py` that are selected only for two- or
  three-channel states. Canonical Navier-Stokes records are single-channel
  vorticity, so those branches do not run. The field is recorded, not applied.
- Only the obstacle Navier-Stokes case adds an interior solid to the geometry
  mask; every other case's geometry is either all zeros (periodic) or the outer
  wall ring.

Geometry is a separate input from the observation mask and the two are never
interchanged. The geometry field describes domain structure; the observation
mask describes sensor placement. This is stated as a contract in
`describe_observations()` (`../src/pdeobs/api/observation.py`) and in
[Observation protocols](observations.md). A sensor count alone does not establish
that all selected sites lie in a fluid region.

## Release tiers

Five tier names select nested prefixes of one canonical record order
(`TIER_SIZES` in `../src/pdeobs/splits.py`, mirrored in
`configs/dataset/default.yaml`):

| Tier | Records per macrodomain |
|---|---|
| `tiny` | 5 |
| `debug` | 20 |
| `signal` | 100 |
| `medium` | 500 |
| `full` | 2,000 |

Every smaller tier is a **subset** of every larger one (`nested_tier_indices`
returns a prefix of one stable order). A tier's records are allocated across the
three regimes by the same largest-remainder rule (`tier_regime_counts`); for the
full tier that is 667 / 667 / 666.

Because the ordering is reconstructible from persisted metadata, a medium-tier
experiment can be taken from existing full-tier shards **without regeneration**:
`_in_release_tier` (`../src/pdeobs/dataset.py`) derives the tier rank as
`3 * regime_sample_index + REGIMES.index(regime)` and keeps records whose rank is
below the tier size.

## Route 1: generate locally

This is the recommended route, and the only route that produces new records.

### One case

```bash
pdeobs generate-case --pde poisson --boundary periodic --setting smooth_grf --param-regime low --num-samples 4 --resolution 16 --seed 17 --root ./my-data
```

Flag defaults, from `build_parser()` in `../src/pdeobs/cli.py`: `--num-samples`
100, `--root` `data`, `--resolution` 128, `--shard-size` 100, `--seed` 20260804;
`--time-steps` is unset by default and a static family rejects any value other
than 1. `--tier` is optional and is inferred from `--num-samples` when it
matches a tier size, otherwise recorded as `custom`. `--dry-run` prints the
resolved jobs without writing records. Shards land under
`<root>/pdeobs_cases/<pde>/<boundary>/<setting>/<regime>/shard_<NNNNN>.h5`, with
the resolved configuration and provenance written beside them under
`_generation/`.

### A campaign

Generation at scale is a two-step, manifest-driven flow:

```bash
pdeobs plan --tier medium --output plan.jsonl
pdeobs generate --plan plan.jsonl --output ./my-data
```

`pdeobs plan` writes a JSONL job manifest plus `plan.resolved.yaml` and
`plan.provenance.json`. `pdeobs generate` executes it. For array execution, one
row (or a bundle of consecutive rows) per array element:

```bash
pdeobs generate --plan plan.jsonl --output ./my-data --array-index "$INDEX" --array-bundle-size 1
```

Generation is resumable per shard. Each shard is written to a sibling
`.partial` file under an exclusive lock and published by a single atomic rename;
on resume a torn partial is repaired by truncating every dataset to the minimum
common row count, and resume matches jobs on a content identity that ignores
volatile execution context (job id, output path, timestamps, scheduler blocks,
runtime host/path fields) while keeping nested solver options
(`../src/pdeobs/storage.py`). A lock file should not be removed merely because it
is old.

### Cost

**Generation cost is unmeasured.** No per-family, per-boundary or per-tier
wall-clock, CPU-hour or core-hour figure for *generating* or *regenerating*
records is recorded anywhere in `configs/` or `docs/`. Regeneration cost is
therefore unmeasured in this release and should not be inferred from other
numbers in the tree.

Other compute numbers that do appear in the tree are **not** generation costs and
do not substitute for one: `configs/campaign/core_observation_medium.yaml`
records a `compute_planning` block explicitly labelled
`status: unmeasured_planning_scenario` (theoretical and usable GPU-hours for a
hypothetical training allocation); [Claims and limits](claims_and_limits.md)
records extrapolated 77.2 and 121.8 GPU-hour training figures and states that no
training was performed to produce them; and `configs/dataset/numerics_full_t15.yaml`
mentions a 48-hour queue limit as the reason for its `shard_size`, not as a
measured duration.

## Route 2: read records you already hold

Load by path:

```python
from pdeobs import api

handle = api.load_dataset("./my-data", source="local")
print(handle.sample_ids_sha256)
```

Only HDF5 is a canonical complete record; a non-`.h5`/`.hdf5` path is refused
rather than coerced (`../src/pdeobs/api/data.py`). The returned `DatasetHandle`
carries `sample_ids_sha256`, a SHA-256 over the **newline-joined** `sample_id`
list — `sha256("\n".join(ids).encode("utf-8"))`, not a bare concatenation — which
gives one identity digest for the selected records.

`load_dataset` returns the handle in memory and writes nothing to disk. A
`dataset.json` file is written only by the creation entry points
(`api.create_dataset` and `api.create_dataset_from_arrays`, which call
`handle.save(out_dir / "dataset.json")`); to persist a handle obtained from
`load_dataset`, call `handle.save(...)` explicitly.

To pin an exact identity set, record the shards you hold and then load through
that manifest:

```bash
pdeobs identity-manifest --data-root ./my-data --shards a/shard_00000.h5 b/shard_00000.h5 --output identity_manifest.json
```

```python
handle = api.load_dataset("./my-data", source="manifest", manifest="identity_manifest.json")
```

The manifest is schema `pdeobs-identity-manifest-v1` and holds `expected_ids`
plus `shards: [{path, sha256}]`. Loading re-hashes **every** listed shard and
raises on any difference before any array is read; absolute or escaping relative
paths are rejected, and a manifest selection cannot be combined with
`split` / `filters` / `max_samples` / `release_tier` shortcuts
(`../src/pdeobs/strict_inference.py`).

## Route 3: download

- **There is no default endpoint.** `DEFAULT_RELEASE_MANIFEST_URL` is `None`,
  with the recorded reason that no anonymous endpoint has been published or
  verified and that the release must not silently contact an author-owned
  service (`../src/pdeobs/download.py`).
- **Omitting `--manifest` fails before any network request.** `pdeobs download`
  raises `Supply an explicit local or HTTPS release manifest with --manifest`
  in the command handler, before the downloader is reached
  (`../src/pdeobs/cli.py`).
- **The Python API refuses a download source** rather than regenerating data
  under a download label: `api.load_dataset(..., source="download")` raises and
  points the caller at local generation or at records they already hold
  (`../src/pdeobs/api/data.py`).
- **When a manifest is supplied, it is enforced.** Schema, status, tiers and
  per-file digests are validated before anything is fetched; a size mismatch
  raises and retains the partial file for resumption, while a checksum mismatch
  deletes the partial file and raises. Transfers resume through an HTTP `Range`
  request, manifest paths that escape the destination root are rejected, and the
  verified manifest is copied to `release-manifest.json` in the destination.

### Release manifest v1 contract

So that a future manifest can be written and checked, the fields enforced by
`validate_release_manifest` are:

| Field | Requirement |
|---|---|
| `schema_version` | required; must equal the integer `1` |
| `name` | required; non-empty string |
| `status` | **optional**; when present must be `gated` or `published`; **an absent `status` defaults to `published`**, the permissive value. `gated` aborts with the optional `gating_reason` in the message |
| `tiers` | required; non-empty list, unique, a subset of `tiny`, `debug`, `signal`, `medium`, `full` |
| `files` | required; non-empty list of objects |
| `files[i].path` | required non-empty string; must not escape the destination root |
| `files[i].sha256` | required; exactly 64 hexadecimal characters |
| `files[i].size` | optional; if present, a non-negative integer, and the downloaded byte count must equal it |
| `files[i].url` | optional; when absent the path is resolved against the manifest's own location |
| `files[i].tiers` / `files[i].tier` | optional; must reference only tiers declared at the top level (defaults to `full`) |

**No example manifest is checked into this repository**, so the contract above
is enforced only in code and no concrete `files[]` entry can be inspected in
context.

## Integrity and provenance per shard

Each canonical shard is an HDF5 file with exactly four datasets — `condition`,
`trajectory`, `geometry`, `metadata` — and five sidecars derived by replacing the
`.h5` suffix (`shard_sidecars` in `../src/pdeobs/storage.py`):

| Sidecar | Holds |
|---|---|
| `<shard>.manifest.json` | `schema_version`, `status`, `shard`, `sample_count`, `bytes` (the measured file size), `sha256`, the full job `spec`, the three other sidecar filenames, `quality_schema_version`, `quality_summary`, `completed_at` |
| `<shard>.sha256` | one `sha256sum`-style line: `"<64-hex>  <shard filename>"`, over the `.h5` payload only |
| `<shard>.metadata.json` | `schema_version`, `shard`, the full per-record metadata list, `quality_summary` |
| `<shard>.metadata.csv` | the same metadata flattened; header is the sorted union of keys, nested values JSON-encoded per cell |
| `<shard>.quality.json` | `schema_version`, `shard`, `sample_count`, the aggregated `quality` summary |

A sixth file, `<shard>.quality-failures.jsonl`, is written only when the
generation quality contract rejects a record, before the failure is raised.
Transient `<shard>.h5.partial` and `<shard>.h5.lock` files exist only while a
shard is being written.

**Completion rule.** `is_shard_complete` treats a shard as complete only if the
file exists and, from its manifest: the manifest schema version matches, the
sample count matches the expected count, the recorded byte size equals the
file's actual size, and (by default) the recomputed SHA-256 equals the manifest
value. When the job spec declares a quality contract, completion additionally
requires the `.quality.json` sidecar to exist, to declare the current quality
schema version, and to carry a quality mapping; publication is refused outright
if per-sample quality coverage is incomplete.

**Sidecar recovery.** Sidecars are rebuilt by re-running the same job with resume
enabled, which is the default (`resume=not args.force` in the `generate` and
`generate-case` handlers). When a finalized `.h5` already exists and validates
against the requested job spec, the writer republishes all five sidecars from it
and returns without recomputing any record (`../src/pdeobs/storage.py`). There is
no standalone recovery subcommand, and no separate tool inspects or clears a
stale lock file.

**Gap.** The `.sha256` file covers the `.h5` payload only. The sidecars
themselves carry no digest and no signature, so manifest, metadata and quality
files are not integrity-protected by anything in this release.

## What a record carries

Per-record metadata is a UTF-8 JSON object in the `metadata` dataset. An
identity is self-describing:

```text
sample_id = seed-<seed>/<pde>/<boundary>/<setting>/<regime>/<6-digit index>
```

for example `seed-20260804/poisson/dirichlet/smooth_grf/low/000042`. The trailing
index is the position within the regime.

| Group | Fields |
|---|---|
| Identity and schema | `sample_id`, `schema_version` (`"1.0"`) |
| Factorial coordinates | `pde`, `boundary`, `setting`, `regime` |
| State | `state_representation` — see the exact value set below |
| Solver provenance | `solver_fidelity`, `solver_version`, `solver_implementation`, `solver_validation_evidence`, `pdeobs_version` |
| Grid and time | `resolution` (`[H, W]`), `T`, `quality_T`, `quality_source`, `stored_frame_indices`, `stored_time_values` (dynamic records with a recorded final time) |
| Position | `regime_sample_index`, `macro_sample_index` |
| Split and tier | `split`, `tier` |
| Seeds | `seed` (derived per-sample), `generation_seed` (campaign base seed) |
| Config provenance | `config_hash`, `git_commit` |
| Official OOD flags | `boundary_ood`, `setting_ood`, `parameter_ood`, `combination_ood` |
| Embedded quality record | `quality` — the full per-sample record, schema `1.3`, including `profile`, `operator` / `operator_id`, `calibration_key` and `calibration_context`, `resolution`, `stored_dtype`, `active_spatial_cells`, `geometry_protocol_id`, a `pde_loss` block, `metrics`, `checks`, `thresholds`, `status` and a `solver` block |

**`state_representation` takes four values, not two.** Filtering on
`state_representation == "vorticity"` silently drops three quarters of the
Navier-Stokes records:

| Records | Stored value |
|---|---|
| All six non-Navier-Stokes families, every boundary | `scalar` |
| Navier-Stokes, `periodic` | `vorticity` |
| Navier-Stokes, `dirichlet` and `neumann` | `bounded_vorticity` |
| Navier-Stokes, `robin_obstacle` | `bounded_obstacle_vorticity` |

All four Navier-Stokes values denote a single scalar vorticity channel
(`../src/pdeobs/pdes/navier_stokes.py`, `../src/pdeobs/generation.py`). To select
across them, request the canonical view rather than matching the raw string:
`_canonical_state_representation` in `../src/pdeobs/dataset.py` collapses the
three solver-specific tags onto `vorticity`, which is what a dataset-level
`state_representation: vorticity` setting asks for.

Per-sample seeds are derived with a stable **personalized** BLAKE2b digest —
`blake2b(digest_size=8, person=b"pdeobs-v1")`, with no key and therefore no
secret — over the base seed and the (family, boundary, setting, regime, index)
tuple, rather than with Python's process-randomized `hash()` (`derive_seed` in
`../src/pdeobs/schema.py`, called from `GenerationJob.sample_seed` in
`../src/pdeobs/generation.py`). Observation masks are reproducible from the
identity and the protocol without re-invoking the solver; see
[Observation protocols](observations.md).

Which of these fields a model is allowed to read is a separate question, answered
in [Tasks and information permissions](tasks_and_permissions.md). Presence in a
Python batch is not permission to use a field as a predictor input.

## Distribution status and terms

| Question | Status in this release |
|---|---|
| Is the full dataset included in this source tree? | No. `release/candidate_metadata.json` records `"full_dataset_included": false`; the corpus is published separately, see [Public release](public_release.md) |
| Has a public upload been verified? | Yes. 3,360 shards / 244 GB at `huggingface.co/datasets/PDE-OBS/pdeobs-data` (re-published 2026-09-26 as a de-identified re-emission: provenance fields replaced, numerical arrays bit-identical); every published file was checked against the staged digests, and `results/public_deposits/release_map.json` binds the generation-time digests to the published ones (the public scrub manifest lists published digests only) |
| Is there a download URL? | Yes, through two manifests under `https://huggingface.co/datasets/PDE-OBS/pdeobs-data/resolve/main/`: `release_manifest.json` (the paper-evaluated slice, 6.5 GB) and `release_manifest_full.json` (the complete corpus, 244 GB), each listing its shards and verification sidecars with the published digests (files live under `data/`). The code still has no default endpoint; `--manifest` stays mandatory |
| Are production checkpoints included in this source tree? | No. `"production_checkpoints_included": false`; all 441 credited checkpoints are published at `huggingface.co/PDE-OBS/pdeobs-models` |
| Does the code licence cover the data? | No. `"code_license_automatically_licenses_external_data": false`; `../LICENSE` covers code only |
| Dataset terms of use? | Declared at deposit. No dataset licence is declared in this tree, and the code licence does not cover data |
| DOI or archive identifier? | None. No DOI or archive identifier is claimed; the deposits are addressed by the Hugging Face locations above |

### Availability commitment

**The complete corpus is deposited publicly.** The deposit is the frozen release measured above,
3,360 shards, published with its checksum manifest and the generation provenance sidecars, so that a
downloaded shard can be verified against the manifest and traced to the case specification that
produced it. The download route already implemented here (`pdeobs download --manifest`) is the
consumer of that manifest; it still has no default endpoint and fails rather than inventing one.
[Public release](public_release.md) gives the locations, the verification result and what the deposit
does not cover.

The deposit exists and was reached and checked without credentials on 26 September 2026
([Public release](public_release.md), `../results/revision_v2/access_check.json`); local regeneration
remains a supported route and rebuilds the same corpus from the recorded seeds and resolved
configuration. The dataset licence is declared at deposit and is not implied by the code licence.

For double-blind review, the deposits are anonymized, reviewer-visible records: links that do not
identify authors, institutions or author-owned infrastructure (a neutral account, cluster labels
`cluster-A/B/C`, de-identified records, one-commit histories). This card is the single place where
such URLs are kept, so that one edit swaps review links for final ones without touching the rest of
the documentation.

| Link | Value in this release |
|---|---|
| Dataset archive (review) | `https://huggingface.co/datasets/PDE-OBS/pdeobs-data` (3,360 shards with sidecars, de-identified re-emission; every file checked against the staged digests on 2026-09-26) |
| Checkpoint archive (review) | `https://huggingface.co/PDE-OBS/pdeobs-models` (441 weights-only checkpoints with de-identified records; checked the same day) |
| Release manifest URL | `https://huggingface.co/datasets/PDE-OBS/pdeobs-data/resolve/main/release_manifest.json` (paper slice) and `.../release_manifest_full.json` (complete corpus); `--manifest` must still be supplied explicitly, the code has no default endpoint |
| Dataset archive (final) | not yet designated; the review deposits above are the current record |
| DOI | none |

## Known gaps in this card

- **No per-family or per-boundary byte breakdown is recorded.** Two
  design-level totals are measured, under "Regenerating the corpus" above:
  3,360 shards / 235.1 GiB for the full factorial and 84 shards / 6.4 GiB for
  the paper slice. Below those, no per-family, per-boundary or per-setting
  corpus size is measured anywhere in the tree, and the byte sizes that do
  exist are per-file or per-run:
  every shard manifest records the measured size of its own `.h5`
  (`bytes`, from the file's own stat, `../src/pdeobs/storage.py`), and the
  acceptance reports carry peak-memory figures rather than storage figures. No
  number is invented here to fill the gap.
- **No example release manifest is checked in**, so the v1 contract above can be
  read only from `../src/pdeobs/download.py`.
- **The paper slice's identity lists are not shipped.** The 1,800 / 200 partition
  is reproducible from `one_setting.stable_split` (`configs/paper/protocol.yaml`),
  but no split receipt and no identity manifest for it is present in this release.
- **No generation or regeneration cost is recorded** for any family or tier (see
  Route 1).
- **No standalone sidecar-recovery command exists.** Re-running the job with
  resume enabled does rebuild all five sidecars from an existing valid `.h5`
  (see "Sidecar recovery" above), but that behaviour is a side effect of resume
  rather than a documented recovery procedure with its own subcommand, and
  nothing inspects or clears a stale lock file.
- **The per-sample quality record schema (1.3) is defined only in code**
  (`QUALITY_SCHEMA_VERSION` in `../src/pdeobs/quality.py`); no document lists its
  fields or the meaning of its `status` and `checks` values.

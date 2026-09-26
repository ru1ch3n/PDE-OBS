# Physical records, identities, and storage

## In-memory representation

`pdeobs.schema.Sample` validates one complete physical record before it enters
storage. All arrays are channel-last:

| Field | Shape of one record | Meaning |
|---|---|---|
| `condition` | `[H,W,Cc]` | PDE-specific coefficient, source, or initial condition |
| `trajectory` | `[T,H,W,Cs]` | Complete solution states; static records have `T=1` |
| `geometry` | `[H,W,Cg]` | Geometry information; built-in generators use one channel |
| `metadata` | JSON-compatible mapping | Physical factors, identity, generation and solver context |

Scalar HW and THW inputs can be normalized to the canonical singleton-channel
form. `Sample` rejects incompatible shapes and non-finite arrays. A geometry
array is not an observation mask: domain structure and sensor placement serve
different purposes. The active scalar/vorticity generator channels must not be
interpreted as a vector velocity simply because two-dimensional spatial grids
are used.

## HDF5 layout

Each shard adds a leading sample axis and stores:

```text
condition    [N,H,W,Cc]
trajectory   [N,T,H,W,Cs]
geometry     [N,H,W,Cg]
metadata     [N] UTF-8 JSON strings
```

Header attributes include schema and generation-specification information.
Metadata carries `sample_id`, the PDE/boundary/setting/regime, seed, and applicable
solver/time context. Different families can have different numbers of stored
times or channels; a single shard enforces consistent array shapes. Use the
metadata rather than guessing a physical parameter from a filename.

```python
from pathlib import Path
from pdeobs.storage import LazyHDF5Dataset

with LazyHDF5Dataset(sorted(Path("my-data").rglob("*.h5")), verify=True) as data:
    sample = data[0]
    print(sample.condition.shape, sample.trajectory.shape)
    print(sample.metadata["sample_id"])
```

Verification depends on the accompanying completion/hash sidecars. A missing
sidecar is not repaired by changing a declared split or pretending a partial
file is complete.

## Deterministic identity and splitting

`GenerationSpec` contains family, boundary, setting, regime, resolution, sample
count, seed, time/storage choices, and content-defining options. `derive_seed`
uses a stable hash rather than Python's process-randomized `hash()`.
The same observation can be reproduced from the sample identity and protocol
seed without resolving the PDE.

Generic dataset split metadata and the paper's one-setting split are not
interchangeable. The paper selects exactly 2,000 records for each fixed
PDE/boundary/setting macrodomain. `one_setting.stable_split` ranks identities by
SHA-256 **within each regime** and uses largest-remainder allocation for exactly
1,800 train and 200 held-out test identities. There is no validation split in
that protocol. The same partition is used across methods and views. Save the
split receipt and actual identity lists; do not infer the paper split from a
generic `split: iid` label alone.

Tiny demos declare their own identity sets. E5 requires disjoint train/test
identities and a separate loading process. No claim that a four-record demo is
the original 200-record paper test set is made.

## Write, resume, and completion

`AtomicHDF5ShardWriter` uses an exclusive creation lock and a `.partial` file.
It appends consistent arrays and metadata, flushes, and finalizes the HDF5 file
before publishing the associated completion information. Sidecars include
manifest, hash, metadata, and quality information.

The finalized HDF5 rename is atomic within its filesystem. Finalizing all
sidecars is a recoverable sequence, not one atomic multi-file transaction.
Interrupted generation may leave a partial file or a finalized file whose
sidecars need recovery. Resume checks content-defining generation fields;
completion is not established by a filename alone. Do not remove a lock merely
because it is old, and do not overwrite an earlier dataset with a scientifically
different configuration under the same identity.

## Observation and task data

`BenchmarkDataset` reads a complete record and constructs a task view at access
time. It can return observations, mask, target, geometry, metadata, and explicitly
declared training-only context. Presence in a Python batch does not mean every
field is an allowed predictor input. The allowed paths are explained in
[Tasks and permissions](tasks_and_permissions.md).

The complete record is the source of supervised targets. Hiding observations
does not remove the ground truth from the storage format; information control
is enforced at the task/model interface and must be checked in external plugins.

## Prediction bundles and result identity

The new strict path consumes a raw prediction bundle plus a versioned contract.
It verifies prediction/target identities, time indices, shapes, and finiteness
before scoring. Bundles are separate from complete physical records and should
not be confused with HDF5 training shards. A result identifies its observation,
task, checkpoint, expected/actual identity counts, and scoring version. See
[Scoring](scoring.md).

Keep historical model checkpoints and score records under their original
attempt/version. New source packaging or stricter evaluation changes must get a
new release/scoring identifier; they do not rewrite historical provenance.

## Dataset size and access status

The full factorial design has seven families, four boundary protocols, ten
condition settings, and 2,000 records per macrodomain: 560,000 planned physical
records, with the 2,000 divided across regimes rather than multiplied by three.
Historical whole-corpus QC records are not a fresh readback in this release.
The selected paper slice contains seven macrodomains, or 14,000 underlying
physical records. Nine observations are not nine new physical solutions.

This candidate does not contain the full corpus or claim its public download has
been verified. A code license is not a blanket dataset license. Small demo data
can be generated from the included code without needing an author machine or
an unverified remote archive.

The candidate downloader requires an explicit manifest; there is no implicit
author-hosted manifest URL. Manifest schema, release state, file checksums and
sizes are checked by the download path. This is downloader functionality, not
evidence that any full dataset endpoint has been published or independently
downloaded in this release.

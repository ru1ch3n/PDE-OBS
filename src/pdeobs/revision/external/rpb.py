"""External data: the CNO authors' Representative PDE Benchmarks (RPB), Poisson part.

Source: Zenodo record 10406879, ``data.zip`` (2,379,635,051 bytes, md5 268a51a9c90f4ee01cd0c1495664ec25), licence CC BY 4.0
(attribution: Raonic et al., Convolutional Neural Operators for robust and accurate learning of PDEs). PDE-OBS does not
redistribute the file; download it from Zenodo and extract ``data/PoissonData_64x64_IN.h5`` and ``data/PoissonData_64x64_OUT.h5``.

Mapping into the V2 record format (no resampling, no value transformation):
    condition  = the stored source "input" f (kept as the PDE condition only; recovery methods never receive it)
    trajectory = the stored solution "output" u (T = 1): the field observed at the sensors and predicted
    geometry   = boundary indicator (outer ring 1), as for the V2 static data
Splits are the authors' (CNO2d_classic/Problems/CNOBenchmarks.py): train = Sample_0..1023, validation = Sample_1024..1151,
test = Sample_1152..1407 of the IN file; ood = Sample_0..255 of the OUT file. One regime ("rpb").

Observation layouts on the native 64^2 Dirichlet grid: U256, C256, U1024, C1024 and G256 exist; the V2 line layouts L256 /
L1024 do not (they need 64 interior columns, the grid has 62) and are refused, never silently replaced. The mixture used for
this source is MXR2_K256 = U256 / C256 under the MXR3 rotation rule (pdeobs.revision.v2masks), not MXR3.

    python -m pdeobs.revision.external.rpb <dir with the two .h5 files> <output V2 store>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

DATASET = "v2_poisson_rpb_ref128"
VIEW = DATASET + "__grid64"
COUNTS = {"train": 1024, "validation": 128, "test": 256, "ood": 256}
FEASIBLE_VIEWS = ("U256", "C256", "U1024", "C1024", "G256")
MIXTURE = "MXR2_K256"
SPLITS = (("train", "PoissonData_64x64_IN.h5", 0, 1024), ("validation", "PoissonData_64x64_IN.h5", 1024, 128),
          ("test", "PoissonData_64x64_IN.h5", 1152, 256), ("ood", "PoissonData_64x64_OUT.h5", 0, 256))
SOURCE = {"record": "zenodo 10406879", "archive": "data.zip", "archive_md5": "268a51a9c90f4ee01cd0c1495664ec25",
          "license": "CC-BY-4.0", "title": "Representative PDE Benchmarks (RPB) - Convolutional Neural Operators"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    import h5py
    import numpy as np
    from ...schema import Sample
    from ...storage import AtomicHDF5ShardWriter
    src_dir = sys.argv[1]
    store = sys.argv[2]
    out_dir = os.path.join(store, VIEW)
    receipt_path = os.path.join(store, "receipts", "rpb", "poisson_grid64.json")
    if os.path.exists(receipt_path) and json.load(open(receipt_path)).get("status") == "completed":
        print(json.dumps({"rpbdata": "already converted", "receipt": receipt_path}))
        return 0
    t0 = time.time()
    n = 64
    geom = np.zeros((n, n, 1), dtype=np.float32)
    geom[[0, -1], :, 0] = 1.0
    geom[:, [0, -1], 0] = 1.0
    rec = {"status": "running", "dataset_id": DATASET, "view": VIEW, "source": dict(SOURCE), "files": {}, "splits": {},
           "mapping": "condition = source input; trajectory[0] = solution output; geometry = boundary ring",
           "values_stored": "float32 copies of the stored arrays (no rescaling)"}
    for fn in sorted({s[1] for s in SPLITS}):
        rec["source"][fn + "_sha256"] = sha256(os.path.join(src_dir, fn))
    for split, fn, start, count in SPLITS:
        path = os.path.join(out_dir, "rpb", split, "part_00000.h5")
        w = AtomicHDF5ShardWriter(path, expected_count=count, spec={"contract": "pdeobs.rev8k.rpb/1", "dataset_id": VIEW,
                                  "split": split, "source_file": fn, "source_range": [start, start + count]},
                                  resume=False, overwrite=True, compression="gzip", compression_opts=4)
        stats = {"u_min": np.inf, "u_max": -np.inf, "f_min": np.inf, "f_max": -np.inf, "u_boundary_absmax": 0.0, "u_rms": []}
        try:
            with h5py.File(os.path.join(src_dir, fn), "r") as f:
                for k in range(start, start + count):
                    g = f["Sample_%d" % k]
                    fin = np.asarray(g["input"][()], dtype=np.float64).reshape(n, n)
                    uout = np.asarray(g["output"][()], dtype=np.float64).reshape(n, n)
                    if not (np.isfinite(fin).all() and np.isfinite(uout).all()):
                        raise ValueError("non-finite values in %s Sample_%d" % (fn, k))
                    stats["u_min"] = min(stats["u_min"], float(uout.min())); stats["u_max"] = max(stats["u_max"], float(uout.max()))
                    stats["f_min"] = min(stats["f_min"], float(fin.min())); stats["f_max"] = max(stats["f_max"], float(fin.max()))
                    b = max(np.abs(uout[[0, -1], :]).max(), np.abs(uout[:, [0, -1]]).max())
                    stats["u_boundary_absmax"] = max(stats["u_boundary_absmax"], float(b))
                    stats["u_rms"].append(float(np.sqrt(np.mean(uout ** 2))))
                    pid = "rpb/poisson/%s/%05d" % ("in" if fn.endswith("_IN.h5") else "out", k)
                    md = {"sample_id": pid + "/rpb/grid64", "parent_id": pid, "dataset_id": VIEW, "schema_version": "1.0",
                          "pde": "poisson", "boundary": "dirichlet", "setting": "rpb_sin_frequency", "regime": "rpb",
                          "ell_code": "rpb", "state_representation": "scalar", "solver_fidelity": "external_reference",
                          "solver_version": "RPB zenodo 10406879", "solver_implementation": "external (CNO authors' release)",
                          "resolution": [n, n], "T": 1, "regime_sample_index": int(k), "split": split, "tier": "rev8k-rpb",
                          "source_file": fn, "source_key": "Sample_%d" % k, "license": "CC-BY-4.0", "parameters": {}}
                    w.append(Sample(fin.astype(np.float32)[..., None], uout.astype(np.float32)[None, ..., None], geom, md))
            man = w.finalize()
        except BaseException:
            w.close()
            raise
        rms = stats.pop("u_rms")
        stats.update(records=count, u_rms_mean=float(np.mean(rms)), u_rms_min=float(np.min(rms)))
        rec["splits"][split] = stats
        rec["files"][split] = {"path": os.path.relpath(path, store), "sha256": man.get("sha256"), "count": int(man.get("sample_count", count))}
    rec.update(status="completed", wall_s=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    with open(receipt_path + ".part", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(receipt_path + ".part", receipt_path)
    print(json.dumps({"rpbdata": rec}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

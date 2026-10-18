"""Rebuild the main tables of the October 2026 revision from the run-level results (results/revision_october).

Inputs:  results/revision_october/runs_long.part-*.csv       one row per (run, evaluation suite, view block)
         results/revision_october/classical_long.part-*.csv  classical references per (task, view, method)
Outputs: <out>/<table>.csv and <out>/tables.md. Standard library only.

Every number is an arithmetic mean over the blocks named in the table (per-record relative L2 averaged over the block's
records by the strict scorer; then averaged over the listed training seeds). "Best neural" is a descriptive envelope over
every trained model of the cohorts listed, not a method selected on test data.

    python reproduce/october/rebuild_tables.py [--results results/revision_october] [--out reproduce/october/out]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

MAIN_SEEDS = ("1101", "1102", "1103")
SLICES = {"ell022": "smooth", "ell009": "medium", "ell0025": "rough", "rpb": "RPB"}
PDE = {"poisson": "Poisson", "darcy": "Darcy", "ns": "NS"}


def parse_id(rid: str) -> dict:
    """FAM__key__ell__model__protocol...__sSEED -> parts; families without a protocol field get it from the family."""
    p = rid.split("__")
    if len(p) < 5 or not p[-1].startswith("s"):
        return {}
    proto = "__".join(p[4:-1]) or {"N3GJ2": "GJ2_K256", "N5MXR6": "MXR6_K256_K1024"}.get(p[0], "")
    return {"family": p[0], "key": p[1], "ell": p[2], "model": p[3], "protocol": proto, "seed": p[-1][1:]}


def load(results: str):
    runs, cls = [], []
    for f in sorted(glob.glob(os.path.join(results, "runs_long.part-*.csv"))):
        runs += list(csv.DictReader(open(f, encoding="utf-8")))
    for f in sorted(glob.glob(os.path.join(results, "classical_long.part-*.csv"))):
        cls += list(csv.DictReader(open(f, encoding="utf-8")))
    for r in runs:
        model_id = r["parent_run_id"] or r["run_id"]
        r["_m"] = parse_id(model_id)
    return runs, cls


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x, nd=3):
    if x is None:
        return "-"
    if x != 0 and abs(x) < 10 ** -nd:
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def seed_mean(values: dict, seeds=MAIN_SEEDS):
    v = [values[s] for s in seeds if values.get(s) is not None]
    return (st.mean(v), len(v)) if v else (None, 0)


def blocks(runs, cohorts, suite):
    """{(family, key, ell, model, protocol): {view: {seed: mean}}} for the main-seed models of the given cohorts."""
    out = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        m = r["_m"]
        if not m or r["suite"] != suite or m["family"] not in cohorts or r["valid"] != "True":
            continue
        out[(m["family"], m["key"], m["ell"], m["model"], m["protocol"])][r["view"]][m["seed"]] = num(r["mean"])
    return out


def classical(cls, task, view, method):
    for c in cls:
        if c["task"] == task and c["view"] == view and c["method"] == method:
            return num(c["mean"])
    return None


def write(out, name, header, rows, md, title, note=""):
    with open(os.path.join(out, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md.append(f"## {title}\n")
    if note:
        md.append(note + "\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "---|" * len(header))
    md.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    md.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/revision_october")
    ap.add_argument("--out", default="reproduce/october/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs, cls = load(a.results)
    md = ["# October 2026 revision: rebuilt tables\n",
          f"Built from {len(runs)} run-level rows and {len(cls)} classical rows. Seeds {', '.join(MAIN_SEEDS)} unless stated.\n"]

    # 1. static calibration (V2 test, 1,800 parents per slice)
    b = blocks(runs, ("V4C", "P4", "P5", "V4D"), "v2eval7")
    rows = []
    for key in ("poisson", "darcy"):
        for ell in ("ell022", "ell009", "ell0025"):
            for v in ("U256", "C256", "L256", "U1024", "C1024", "L1024", "G256"):
                cand = []
                for (fam, k, e, model, proto), byv in b.items():
                    if (k, e) == (key, ell) and v + "__rep0" in byv:
                        m, n = seed_mean(byv[v + "__rep0"])
                        if n == len(MAIN_SEEDS):
                            cand.append((m, f"{fam} {model} {proto}"))
                if not cand:
                    continue
                best = min(cand)
                task = f"V4cls__{key}__{ell}"
                rows.append([PDE[key], SLICES[ell], v, fmt(classical(cls, task, v, "cov_linear"), 4),
                             fmt(classical(cls, task, v, "gappy_pod_ridge")), fmt(classical(cls, task, v, "linear_interp_nearest_fallback")),
                             fmt(best[0]), best[1]])
    write(a.out, "static_calibration", ["PDE", "slice", "view", "covariance", "gappy POD", "interpolation", "best neural (envelope)", "model"],
          rows, md, "1. Static calibration (V2 test)", "Classical references fitted on train, settings chosen on validation.")

    # 2. observation transfer: matched (D3) / off-pattern (C3) per seed, U / C / L specialists
    rows = []
    for (fam, key, ell, model, proto), _ in sorted(b.items()):
        if proto != "U256" or fam not in ("V4C", "P4", "P5"):
            continue
        spec = {p: b.get((fam, key, ell, model, p)) for p in ("U256", "C256", "L256")}
        if not all(spec.values()):
            continue
        per_seed = []
        for s in MAIN_SEEDS:
            try:
                d = st.mean(spec[p][p + "__rep0"][s] for p in spec)
                c = st.mean(spec[p][q + "__rep0"][s] for p in spec for q in spec if q != p)
            except (KeyError, TypeError):
                continue
            per_seed.append((d, c))
        if not per_seed:
            continue
        mx = b.get((fam, key, ell, model, "MXR3_K256"))
        mx3 = seed_mean({s: st.mean(mx[q + "__rep0"][s] for q in spec) for s in MAIN_SEEDS
                         if mx and all(s in mx.get(q + "__rep0", {}) for q in spec)})[0] if mx else None
        rows.append([fam, PDE.get(key, key), SLICES.get(ell, ell), model, fmt(st.mean(x[0] for x in per_seed)),
                     fmt(st.mean(x[1] for x in per_seed)), " / ".join(f"{c / d:.2f}" for d, c in per_seed), fmt(mx3), len(per_seed)])
    write(a.out, "transfer", ["cohort", "PDE", "slice", "model", "D3 matched", "C3 off-pattern", "C3/D3 per seed", "MXR3 on U/C/L", "seeds"],
          rows, md, "2. Observation transfer (V2 test)")

    # 3. grid jitter (Poisson rough), with the data-consistency split where the evaluation recorded it
    rows = []
    for fam_eval in ("N3ev", "E2ev"):
        g = defaultdict(lambda: defaultdict(dict))
        dc = defaultdict(lambda: defaultdict(dict))
        for r in runs:
            if r["cohort"] != fam_eval or r["suite"] != "n3geom" or not r["_m"]:
                continue
            m = r["_m"]
            key = (m["family"], m["model"], m["protocol"])
            g[key][r["view"]][m["seed"]] = num(r["mean"])
            if r["dc_hid"]:
                dc[key][r["view"]][m["seed"]] = num(r["dc_hid"])
        for key in sorted(g):
            byv = g[key]

            def jit(j, src=byv):
                vals = [seed_mean(src[f"GJ{j}_K256_r{q}"])[0] for q in range(3) if f"GJ{j}_K256_r{q}" in src]
                return st.mean(vals) if vals else None
            shift = [seed_mean(byv[x])[0] for x in byv if x.startswith("GT")]
            rows.append([fam_eval, key[1], key[2], fmt(seed_mean(byv.get("G256", {}))[0]), fmt(st.mean(shift) if shift else None),
                         fmt(jit(1)), fmt(jit(2)), fmt(jit(3)), fmt(seed_mean(byv.get("U256", {}))[0]),
                         fmt(seed_mean(dc[key].get("G256", {}))[0]) if dc.get(key) else "-",
                         fmt(jit(3, dc[key])) if dc.get(key) else "-"])
    write(a.out, "grid_jitter", ["evaluation", "model", "trained on", "G256", "coherent shift", "jitter 1", "jitter 2", "jitter 3", "U256",
                                 "hid (G256)", "hid (jitter 3)"],
          rows, md, "3. Grid jitter, Poisson rough (V2 test)", "hid = error of the data-consistent prediction (observed cells set to the observations).")

    # 4. NS medium sensor density (nested uniform sets, three realizations averaged)
    rows = []
    g = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r["cohort"] == "N5ev" and r["suite"] == "n5nested" and r["_m"]:
            m = r["_m"]
            g[(m["family"], m["model"], m["protocol"])][r["view"]][m["seed"]] = num(r["mean"])
    for key in sorted(g):
        row = [key[1], key[2]]
        for k in (64, 128, 256, 512, 1024, 4096):
            vals = [seed_mean(g[key][f"U{k}__q{q}"])[0] for q in range(3) if f"U{k}__q{q}" in g[key]]
            row.append(fmt(st.mean(vals)) if vals else "-")
        rows.append(row)
    write(a.out, "ns_medium_density", ["model", "trained on", "K=64", "128", "256", "512", "1,024", "4,096"], rows, md,
          "4. NS medium sensor density (V2 test)")

    # 5. resolution (N8 trainings at 64^2 / 256^2; 128^2 anchors are the V4C FNO / P4 Voronoi specialists)
    rows = []
    b8 = blocks(runs, ("N8", "V4C", "P4"), "v2eval7")
    for key in ("poisson", "ns"):
        for model, anchor in (("fno", "V4C"), ("voronoi_cnn", "P4")):
            for p in ("U256", "C256"):
                row = [PDE[key], model, p]
                for grid in (64, 128, 256):
                    k = ("N8", key, "ell0025", model, f"{p}__g{grid}") if grid != 128 else (anchor, key, "ell0025", model, p)
                    byv = b8.get(k, {})
                    mm, n = seed_mean(byv.get(p + "__rep0", {}))
                    uu, _ = seed_mean(byv.get("U1024__rep0", {}))
                    row.append(f"{fmt(mm)} / {fmt(uu)}" + (f" ({n} seeds)" if 0 < n < 3 else "") if n else "-")
                rows.append(row)
    write(a.out, "resolution", ["PDE (rough)", "model", "trained on", "64^2 matched / U1024", "128^2", "256^2"], rows, md,
          "5. Resolution (V2 test)")

    # 6. external RPB Poisson: in-distribution test and out-of-distribution, with the classical references
    rows = []
    views = ("U256", "C256", "U1024", "C1024", "G256")
    for pop, suite, fams, ctask in (("ID test", "v2eval7", ("RPB", "RPBn"), "V4cls__poisson__rpb"),
                                    ("OOD", "rpbood", ("RPB", "RPBn"), "V4cls__poisson__rpb__ood")):
        g = defaultdict(lambda: defaultdict(dict))
        for r in runs:
            m = r["_m"]
            if not m or r["suite"] != suite or m["family"] not in fams or r["cohort"] in ("E2ev",):
                continue
            g[(m["family"], m["model"], m["protocol"])][r["view"].split("__")[0]][m["seed"]] = num(r["mean"])
        for key in sorted(g):
            lab = f"{key[1]}{' + normaliser' if key[0] == 'RPBn' else ''}"
            rows.append([pop, lab, key[2]] + [fmt(seed_mean(g[key].get(v, {}))[0]) for v in views])
        for meth, lab in (("cov_linear", "covariance"), ("gappy_pod_ridge", "gappy POD"), ("linear_interp_nearest_fallback", "interpolation")):
            rows.append([pop, lab, "-"] + [fmt(classical(cls, ctask, v, meth), 4) for v in views])
    write(a.out, "rpb", ["population", "method", "trained on"] + list(views), rows, md, "6. External RPB Poisson",
          "Frozen-recipe Voronoi-CNN runs failed (training pathology) and have no rows; '+ normaliser' is the single-factor control.")

    # 7. normalisation-only attribution (one change at a time; per test view)
    rows = []
    bn = blocks(runs, ("N6", "V4C", "P5", "P4"), "v2eval7")
    for fam, key, ell, model, proto, lab in (
            ("P5", "poisson", "ell009", "cno_classic", "MXR3_K256", "CNO Poisson medium: frozen recipe (LR 1e-3, no normaliser)"),
            ("N6", "poisson", "ell009", "cno_classic", "MXR3_K256__nonorm", "CNO Poisson medium: V4 learning rate, normaliser off"),
            ("V4C", "poisson", "ell009", "cno_classic", "MXR3_K256", "CNO Poisson medium: V4 recipe"),
            ("P4", "darcy", "ell0025", "fno", "MXR3_K256", "FNO Darcy rough: no normaliser (frozen P4)"),
            ("V4C", "darcy", "ell0025", "fno", "MXR3_K256", "FNO Darcy rough: V4 recipe")):
        byv = bn.get((fam, key, ell, model, proto), {})
        rows.append([lab] + [fmt(seed_mean(byv.get(v + "__rep0", {}))[0]) for v in ("U256", "C256", "L256", "U1024")])
    write(a.out, "normalisation", ["model / recipe (MXR3-trained)", "U256", "C256", "L256", "U1024"], rows, md,
          "7. Normalisation-only attribution")

    # 8. confirmation set (neural models; 600 sealed parents per slice)
    rows = []
    gc = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r["cohort"] == "N7ev" and r["_m"]:
            m = r["_m"]
            gc[(m["key"], m["ell"])][(m["family"], m["model"], m["protocol"], r["view"])][m["seed"]] = num(r["mean"])
    for (key, ell), d in sorted(gc.items()):
        for v in ("U256__q0", "C256__q0", "L256__q0", "U1024__q0"):
            cand = [(seed_mean(s)[0], f"{f} {mo} {p}") for (f, mo, p, vv), s in d.items() if vv == v and seed_mean(s)[1] == 3]
            if cand:
                best = min(cand)
                rows.append([PDE.get(key, key), SLICES.get(ell, ell), v.split("__")[0], fmt(best[0]), best[1]])
    write(a.out, "confirmation_neural", ["PDE", "slice", "view", "best neural (envelope)", "model"], rows, md,
          "8. Confirmation set, neural models",
          "The classical confirmation scores are in the unified report; they are not part of this run-level export.")

    # 9. failures (kept as results, never replaced)
    rows = []
    for r in runs:
        if r["run_status"].startswith("failed"):
            rows.append([r["run_id"], r["run_status"].split(":", 2)[-1].strip()[:80]])
    write(a.out, "failures", ["run", "reason"], sorted(rows), md, "9. Training failures")
    open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print(f"{len(runs)} rows -> {a.out}/tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

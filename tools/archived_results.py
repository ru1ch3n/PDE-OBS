"""Index, check and tabulate ARCHIVED campaign results without re-running anything.

The paper's runs were produced by the campaign runner, whose outputs have this layout:

    <training root>/<wave>/<pde>/<method>/<train view>/
        identity.json  completion.json  health.json  split_manifest.json  factor_coverage.json
        budget-protocol.json | dynamic-protocol.json | operational-continuation*.json   (cohort dependent)
        history.json  provenance.json  resolved.yaml  checkpoints/{last.pt, best.pt, training_config.json}
    <evaluation root>/<wave>/<sha256 or pde/method/view[_suffix]>/
        completion.json  blocks/<test view>/metrics.json   (nine test views, 200 identities each)

No prediction tensors were retained by that runner, so these results carry the frozen campaign
evaluator's metrics (the legacy relative-L2 family), never `pdeobs-strict-v1` blocks.  This tool does
not change that: it reads what exists, hashes it, links evaluation to training by checkpoint identity,
classifies the training cohort from the recorded protocol fields, checks the links, and emits the
long-format results table plus the matched-diagonal and cross-view pivots described in
docs/results_format.md.  Every table row carries its cohort, actual epochs, stop reason, checkpoint
identity, attempt identity and scoring version, and a missing cell is an explicit marker, never a number.

Three verbs, all read-only with respect to the archives:

  scan   run on the machine that holds the archives; walks both roots and writes one JSON file
         (JSON records parsed, every file hashed; no tensors, no history bodies)
  pair   joins one or more scan files: evaluation <-> training by checkpoint SHA-256 (or by the
         source-artifact hashes a salvage evaluation records), one selected attempt per identity
  index  writes the public index, the checks and the tables into a NEW directory

Python 3.6+ standard library only, so it runs on cluster login nodes as they are.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys

SCAN_VERSION = "pdeobs-archived-results-scan/v1"
PAIR_VERSION = "pdeobs-archived-results-pair/v1"
INDEX_VERSION = "pdeobs-archived-results-index/v1"
VIEWS = ["random_50pct", "random_65pct", "random_80pct", "block_observed_50pct", "line_sensors_50pct",
         "horizontal_lines_50pct", "vertical_lines_50pct", "boundary_band_50pct", "clustered_50pct"]
PDES = ["darcy", "poisson", "helmholtz", "heat", "reaction_diffusion", "burgers", "navier_stokes"]
STATIC_PDES = {"darcy", "poisson", "helmholtz"}
METHODS = ["ufno_2d", "fno", "cno", "deeponet", "gnot", "transolver", "pino"]
TRAIN_RECORDS = ["identity.json", "completion.json", "health.json", "split_manifest.json", "factor_coverage.json",
                 "budget-protocol.json", "dynamic-protocol.json", "checkpoints/training_config.json", "provenance.json"]
HASH_ONLY = ["history.json", "resolved.yaml"]
SPLIT_ALGORITHM = "sha256_rank_within_regime_largest_remainder_exact_ten_percent_test"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MISSING = "RESULT_PENDING"
TRAIN_CONFIG_PUBLIC = ["epochs", "optimizer", "learning_rate", "weight_decay", "scheduler", "scheduler_step_size", "scheduler_gamma",
                       "scheduler_pct_start", "scheduler_div_factor", "scheduler_final_div_factor", "scheduler_anneal_strategy",
                       "loss", "data_loss_weight", "physics_loss", "physics_loss_weight", "grad_clip", "gradient_norm_warning",
                       "gradient_warning_fatal", "health_policy", "deterministic", "amp", "horizon", "history_steps",
                       "rollout_target_offset", "target_step", "teacher_forcing_ratio", "batch_size", "seed", "task",
                       "output_std_ratio_collapse_max", "collapse_consecutive_windows", "early_stopping_patience", "monitor"]


# ----------------------------------------------------------------------------- helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        f.write("\n")


def relpath(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def identity_of(record):
    pde, method, view = record.get("pde"), record.get("method"), record.get("training_view")
    if pde and method and view:
        return "%s/%s/%s" % (pde, method, view)
    return None


def task_of(pde):
    return "recovery" if pde in STATIC_PDES else "rollout"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ----------------------------------------------------------------------------- scan

def scan_evaluation(root):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "completion.json" not in filenames:
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= 5:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            continue
        dirnames[:] = []
        entry = {"rel": relpath(dirpath, root), "files": {}, "blocks": {}}
        cpath = os.path.join(dirpath, "completion.json")
        try:
            entry["completion"] = read_json(cpath)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            entry["completion"] = None
            entry["error"] = "completion.json unreadable: %s" % exc
        entry["files"]["completion.json"] = {"sha256": sha256_file(cpath), "bytes": os.path.getsize(cpath)}
        bdir = os.path.join(dirpath, "blocks")
        if os.path.isdir(bdir):
            for view in sorted(os.listdir(bdir)):
                mpath = os.path.join(bdir, view, "metrics.json")
                if not os.path.isfile(mpath):
                    continue
                rel_m = "blocks/%s/metrics.json" % view
                entry["files"][rel_m] = {"sha256": sha256_file(mpath), "bytes": os.path.getsize(mpath)}
                try:
                    m = read_json(mpath)
                except Exception as exc:  # noqa: BLE001
                    entry["blocks"][view] = {"error": "metrics.json unreadable: %s" % exc}
                    continue
                cfg = m.get("evaluation_config") or {}
                entry["blocks"][view] = {
                    "metrics": m.get("metrics"), "samples": m.get("samples"), "checkpoint_sha256": m.get("checkpoint_sha256"),
                    "test_sample_ids_sha256": m.get("test_sample_ids_sha256"), "task": m.get("task"), "config_hash": m.get("config_hash"),
                    "evaluation_view": m.get("evaluation_view"), "training_view": m.get("training_view"), "pde": m.get("pde"),
                    "method": m.get("method"), "mask_protocol": m.get("mask_protocol"), "observation_ratio": m.get("observation_ratio"),
                    "observation_mode": m.get("observation_mode"), "split": m.get("split"),
                    "evaluation_config": {k: cfg.get(k) for k in ("task", "horizon", "horizons", "history_steps", "rollout_target_offset",
                                                                  "target_step", "observation_mode", "physical_representation")},
                }
        entries.append(entry)
    return entries


def scan_training(root, hash_checkpoints=False):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        is_output = any(n in filenames for n in ("identity.json", "health.json", "completion.json"))
        if not is_output:
            if depth >= 4:
                dirnames[:] = []
            continue
        dirnames[:] = []
        entry = {"rel": relpath(dirpath, root), "files": {}, "records": {}, "checkpoints": {}, "continuations": []}
        for name in TRAIN_RECORDS + HASH_ONLY:
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            entry["files"][name] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
            if name in HASH_ONLY:
                continue
            try:
                entry["records"][name] = read_json(path)
            except Exception as exc:  # noqa: BLE001
                entry["records"][name] = {"_error": "unreadable: %s" % exc}
        for fn in sorted(filenames):
            if fn.startswith("operational-continuation") and fn.endswith(".json"):
                path = os.path.join(dirpath, fn)
                entry["files"][fn] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
                try:
                    entry["continuations"].append({"file": fn, "record": read_json(path)})
                except Exception as exc:  # noqa: BLE001
                    entry["continuations"].append({"file": fn, "record": {"_error": str(exc)}})
        cdir = os.path.join(dirpath, "checkpoints")
        if os.path.isdir(cdir):
            for fn in sorted(os.listdir(cdir)):
                if fn.endswith(".pt"):
                    path = os.path.join(cdir, fn)
                    info = {"bytes": os.path.getsize(path)}
                    if hash_checkpoints:
                        info["sha256"] = sha256_file(path)
                    entry["checkpoints"][fn] = info
            for sub in ("budget200-state", "continuation-state", "dynamic-phase-state"):
                sdir = os.path.join(cdir, sub)
                if os.path.isdir(sdir):
                    entry["checkpoints"][sub + "/"] = {"count": len([x for x in os.listdir(sdir) if x.endswith(".json")])}
        entry["lock_present"] = os.path.exists(dirpath + ".lock")
        entries.append(entry)
    return entries


def cmd_scan(args):
    payload = {"schema_version": SCAN_VERSION, "label": args.label,
               "roots": {"training": args.training_root, "evaluation": args.evaluation_root},
               "hash_checkpoints": bool(args.hash_checkpoints)}
    payload["evaluation"] = scan_evaluation(args.evaluation_root) if args.evaluation_root else []
    payload["training"] = scan_training(args.training_root, args.hash_checkpoints) if args.training_root else []
    payload["counts"] = {"evaluation_outputs": len(payload["evaluation"]), "training_outputs": len(payload["training"])}
    if args.out == "-":
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
    else:
        if os.path.exists(args.out):
            sys.stderr.write("refusing to overwrite %s\n" % args.out)
            return 2
        write_json(args.out, payload)
        sys.stderr.write("scan: %s\n" % json.dumps(payload["counts"]))
    return 0


# ----------------------------------------------------------------------------- pair

def training_key_maps(trainings):
    by_ckpt, by_filehash, by_identity = {}, {}, {}
    for t in trainings:
        rec = t["records"]
        comp = rec.get("completion.json") or {}
        ck = comp.get("checkpoint_sha256")
        if ck:
            by_ckpt.setdefault(ck, []).append(t)
        for fname, info in t["files"].items():
            by_filehash.setdefault(info["sha256"], []).append((fname, t))
        for fn, info in t.get("checkpoints", {}).items():
            if info.get("sha256"):
                by_ckpt.setdefault(info["sha256"], []).append(t)
        ident = identity_of(rec.get("identity.json") or comp or {})
        if ident:
            by_identity.setdefault(ident, []).append(t)
    return by_ckpt, by_filehash, by_identity


def pair_one(eva, maps):
    by_ckpt, by_filehash, by_identity = maps
    comp = eva.get("completion") or {}
    ck = comp.get("checkpoint_sha256") or (comp.get("policy_evidence") or {}).get("checkpoint_sha256")
    how = None
    cands = []
    if ck and ck in by_ckpt:
        cands, how = by_ckpt[ck], "checkpoint_sha256"
    if not cands:
        src = (comp.get("policy_evidence") or {}).get("source_artifact_sha256") or {}
        for fname in ("completion.json", "health.json", "identity.json", "budget-protocol.json"):
            h = src.get(fname)
            if h and h in by_filehash:
                cands = [t for (fn, t) in by_filehash[h] if fn == fname]
                if cands:
                    how = "source_artifact_sha256:" + fname
                    break
    if not cands:
        ident = identity_of(comp)
        same = by_identity.get(ident, []) if ident else []
        if len(same) == 1:
            cands, how = same, "unique_training_for_identity"
    uniq = []
    for c in cands:
        if c["host"] == eva["host"] and c not in uniq:
            uniq.append(c)
    if not uniq:
        uniq = cands
    if len(uniq) > 1:
        # identical training records on two archives (a mirrored copy) describe one attempt
        keys = {json.dumps({k: v.get("sha256") for k, v in c["files"].items() if k in ("completion.json", "identity.json", "health.json")}, sort_keys=True) for c in uniq}
        if len(keys) == 1:
            uniq, how = uniq[:1], (how or "") + "+mirrored_copy"
    return uniq, how


def cmd_pair(args):
    scans = [read_json(p) for p in args.scan]
    evas, trains = [], []
    for s in scans:
        if s.get("schema_version") != SCAN_VERSION:
            sys.stderr.write("not a scan file: %s\n" % s.get("schema_version"))
            return 2
        label = s.get("label") or "scan%d" % len(evas)
        for e in s["evaluation"]:
            e["host"] = label
            evas.append(e)
        for t in s["training"]:
            t["host"] = label
            trains.append(t)
    maps = training_key_maps(trains)
    selection = read_json(args.selection) if args.selection else {}
    groups = {}
    for e in evas:
        comp = e.get("completion") or {}
        ident = identity_of(comp)
        if not ident:
            ident = "unidentified/%s/%s" % (e["host"], e["rel"])
        cands, how = pair_one(e, maps)
        e["training_candidates"] = [{"host": c["host"], "rel": c["rel"]} for c in cands]
        e["training"] = cands[0] if len(cands) == 1 else None
        e["pairing"] = how if len(cands) == 1 else ("ambiguous:%d" % len(cands) if cands else "unpaired")
        groups.setdefault(ident, []).append(e)
    paired = {"schema_version": PAIR_VERSION, "scans": [s.get("label") for s in scans], "identities": {}, "unmatched_training": []}
    used = set()
    for ident, cands in sorted(groups.items()):
        chosen, rule = select_attempt(ident, cands, selection.get(ident))
        for c in cands:
            if c.get("training"):
                used.add((c["training"]["host"], c["training"]["rel"]))
        paired["identities"][ident] = {
            "selected": chosen, "selection_rule": rule,
            "alternatives": [{"host": c["host"], "rel": c["rel"], "status": (c.get("completion") or {}).get("status"),
                              "completed_blocks": (c.get("completion") or {}).get("completed_blocks"),
                              "completion_sha256": c["files"]["completion.json"]["sha256"]} for c in cands if c is not chosen],
        }
    for t in trains:
        if (t["host"], t["rel"]) not in used:
            rec = t["records"]
            ident = identity_of(rec.get("identity.json") or rec.get("completion.json") or {})
            h = rec.get("health.json") or {}
            paired["unmatched_training"].append({"identity": ident, "host": t["host"], "rel": t["rel"],
                                                 "status": h.get("status") or (rec.get("completion.json") or {}).get("status"),
                                                 "epochs_completed": h.get("epochs_completed"),
                                                 "events": [ev.get("kind") for ev in (h.get("events") or []) if isinstance(ev, dict)],
                                                 "error": (h.get("error") or "")[:200]})
    if os.path.exists(args.out):
        sys.stderr.write("refusing to overwrite %s\n" % args.out)
        return 2
    write_json(args.out, paired)
    n_sel = sum(1 for v in paired["identities"].values() if v["selected"])
    sys.stderr.write("pair: %d identities with an evaluation, %d selected, %d unmatched training outputs\n"
                     % (len(paired["identities"]), n_sel, len(paired["unmatched_training"])))
    return 0


def select_attempt(ident, cands, wanted):
    """One evaluation per identity. An explicit selection wins; otherwise the unique complete one; else the latest complete one, flagged."""
    complete = [c for c in cands if (c.get("completion") or {}).get("status") == "completed"
                and (c.get("completion") or {}).get("completed_blocks") == (c.get("completion") or {}).get("expected_blocks")
                and len(c.get("blocks") or {}) == 9]
    if wanted:
        for c in cands:
            sha = c["files"]["completion.json"]["sha256"]
            if wanted.get("evaluation_completion_sha256") == sha or (wanted.get("evaluation_rel") and c["rel"].endswith(wanted["evaluation_rel"])):
                return c, "explicit_selection"
        ck = wanted.get("checkpoint_sha256")
        if ck:
            hits = [c for c in complete if (c.get("completion") or {}).get("checkpoint_sha256") == ck]
            if len(hits) == 1:
                return hits[0], "explicit_selection_by_checkpoint"
        return None, "explicit_selection_not_found"
    if len(complete) == 1:
        return complete[0], "unique_complete_evaluation"
    if len(complete) > 1:
        complete.sort(key=lambda c: (c.get("completion") or {}).get("completed_unix") or 0)
        return complete[-1], "latest_of_%d_complete_evaluations_UNVERIFIED_CHOICE" % len(complete)
    return None, "no_complete_evaluation"


# ----------------------------------------------------------------------------- index

def classify_cohort(train, eva_comp):
    """Cohort from the recorded protocol fields; the evidence is returned with the label."""
    rec = (train or {}).get("records", {}) if train else {}
    comp = rec.get("completion.json") or {}
    ident = rec.get("identity.json") or {}
    health = rec.get("health.json") or {}
    proto = comp.get("training_protocol") or ident.get("training_protocol") or (eva_comp.get("training_protocol") if eva_comp else None)
    planned = comp.get("planned_epochs") or ident.get("planned_epochs") or ident.get("epochs")
    done = comp.get("epochs_completed") or comp.get("actual_epochs") or health.get("epochs_completed")
    if eva_comp and eva_comp.get("actual_training_epochs") and not done:
        done = eva_comp.get("actual_training_epochs")
    continuations = bool((train or {}).get("continuations")) or bool(rec.get("dynamic-protocol.json"))
    evidence = {"training_protocol": proto, "planned_epochs": planned, "epochs_completed": done, "stop_reason": comp.get("stop_reason"),
                "training_status": comp.get("status") or health.get("status"), "accepted": comp.get("accepted", health.get("accepted")),
                "continuation_records": [c["file"] for c in (train or {}).get("continuations", [])],
                "budget_protocol_present": "budget-protocol.json" in rec, "dynamic_protocol_present": "dynamic-protocol.json" in rec,
                "patience": comp.get("patience", ident.get("patience")), "patience_start_epoch": comp.get("patience_start_epoch", ident.get("patience_start_epoch")),
                "warmup_complete_epochs": comp.get("warmup_complete_epochs", ident.get("warmup_complete_epochs")), "monitor": comp.get("monitor", ident.get("monitor"))}
    if proto and "max200-min120" in proto:
        label = "budget200_min120"
    elif proto and "max200" in proto:
        label = "budget200_patience10_no_floor"  # not listed in configs/paper/training_cohorts.yaml; reported as recorded
    elif comp.get("status") == "completed" and planned == 500 and done == 500:
        label = "original500" + ("_recovered" if continuations else "")
    elif comp.get("status") == "completed" and planned == 200 and done == 200:
        label = "fixed200" + ("_recovered" if continuations else "")
    elif comp.get("status") == "completed" and planned and done and done >= planned:
        label = "declared_budget_%s_completed" % planned
    else:
        label = "interrupted_or_recovered"
    evidence["budget_complete"] = bool(planned and done and (done >= planned or (proto and "patience" in str(comp.get("stop_reason") or ""))))
    return label, evidence


def public_training_config(rec):
    cfg = rec.get("checkpoints/training_config.json") or {}
    return {k: cfg.get(k) for k in TRAIN_CONFIG_PUBLIC if k in cfg}


def split_check(split, pde):
    d = []
    ok = True
    if not split:
        return "blocked", ["split_manifest.json missing"]
    if split.get("algorithm") != SPLIT_ALGORITHM:
        ok = False; d.append("algorithm %r" % split.get("algorithm"))
    if split.get("seed") != 20260804:
        ok = False; d.append("seed %r" % split.get("seed"))
    if (split.get("records"), split.get("train_records"), split.get("test_records"), split.get("validation_records")) != (2000, 1800, 200, 0):
        ok = False; d.append("counts %r" % [split.get(k) for k in ("records", "train_records", "test_records", "validation_records")])
    rc = split.get("regime_counts") or {}
    trains = sorted(int(v.get("train", -1)) for v in rc.values()) if isinstance(rc, dict) else []
    tests = sorted(int(v.get("test", -1)) for v in rc.values()) if isinstance(rc, dict) else []
    if trains != [600, 600, 600] or tests != [66, 67, 67]:
        ok = False; d.append("regime allocation train %s test %s" % (trains, tests))
    mid = split.get("macrodomain_identity") or ""
    if not (isinstance(mid, str) and mid.count("|") == 2 and mid.startswith(pde + "|")):
        ok = False; d.append("macrodomain_identity %r" % mid)
    if ok:
        d.append("%s: 2000 records, 1800/200, regimes 600/600/600 and 67/67/66, seed 20260804, %s" % (split.get("schema_version"), mid))
    return ("passed" if ok else "failed"), d


def build_record(ident, sel, train):
    comp = sel.get("completion") or {}
    pde, method, train_view = ident.split("/")
    rec = train.get("records", {}) if train else {}
    tcomp = rec.get("completion.json") or {}
    health = rec.get("health.json") or {}
    identity = rec.get("identity.json") or {}
    split = rec.get("split_manifest.json")
    cohort, cohort_evidence = classify_cohort(train, comp)
    blocks = sel.get("blocks") or {}
    checks = {}
    # blocks complete
    d = []
    present = [v for v in VIEWS if v in blocks and blocks[v].get("metrics") is not None]
    missing = [v for v in VIEWS if v not in present]
    samples = sorted({blocks[v].get("samples") for v in present}, key=lambda x: (x is None, x))
    ids = sorted({blocks[v].get("test_sample_ids_sha256") for v in present}, key=lambda x: (x is None, str(x)))
    bad = False
    if missing:
        d.append("missing views: %s" % missing); bad = True
    if samples != [200]:
        d.append("samples per block %s (200 expected)" % samples); bad = True
    if len(ids) != 1 or ids[0] is None:
        d.append("test identity set differs across blocks or is unrecorded: %s" % ids); bad = True
    elif not split or not split.get("test_sample_ids_sha256"):
        d.append("split manifest carries no test_sample_ids_sha256; block identity set not bound to the split"); bad = True
    elif split["test_sample_ids_sha256"] != ids[0]:
        d.append("blocks test_sample_ids_sha256 differs from the split manifest"); bad = True
    if not bad:
        d.append("nine views, 200 identities each, one identity set %s..., equal to the split manifest" % ids[0][:12])
    checks["blocks_complete"] = {"status": "failed" if bad else "passed", "details": d}
    # checkpoint binding
    d = []
    ck_eva = comp.get("checkpoint_sha256")
    ck_blocks = sorted({blocks[v].get("checkpoint_sha256") for v in present})
    ck_train = tcomp.get("checkpoint_sha256")
    ck_file = ((train or {}).get("checkpoints") or {}).get("last.pt", {}).get("sha256") if train else None
    ck_policy = (comp.get("policy_evidence") or {}).get("checkpoint_sha256")
    refs = {"evaluation_completion": ck_eva, "blocks": ck_blocks[0] if len(ck_blocks) == 1 else "MIXED:%s" % ck_blocks,
            "training_completion": ck_train, "policy_evidence": ck_policy, "checkpoint_file": ck_file}
    values = {k: v for k, v in refs.items() if v}
    distinct = set(values.values())
    if not ck_eva:
        d.append("evaluation completion records no checkpoint_sha256")
    if len(distinct) == 1 and ck_eva:
        st = "passed"
        d.append("checkpoint %s... is the same identity in: %s" % (ck_eva[:12], ", ".join(sorted(values))))
        if not ck_file:
            d.append("checkpoint file not re-hashed in this pass (scan without --hash-checkpoints); binding is between records")
    elif len(distinct) > 1:
        st = "failed"; d.append("checkpoint identities disagree: %s" % values)
    else:
        st = "blocked"
    checks["checkpoint_binding"] = {"status": st, "details": d}
    # metrics file binding
    d = []
    cblocks = comp.get("blocks") or {}
    bad, verified = [], 0
    for v in present:
        recorded = None
        if isinstance(cblocks, dict):
            recorded = (cblocks.get(v) or {}).get("metrics_sha256") or (cblocks.get(v) or {}).get("result_sha256")
        elif isinstance(cblocks, list):
            for b in cblocks:
                if b.get("view") == v:
                    recorded = b.get("metrics_sha256") or b.get("result_sha256")
        actual = (sel["files"].get("blocks/%s/metrics.json" % v) or {}).get("sha256")
        if recorded and actual:
            verified += 1
            if recorded != actual:
                bad.append(v)
    if bad:
        st = "failed"; d.append("metrics.json hash differs from the completion record for %s" % bad)
    elif verified:
        st = "passed"; d.append("%d block metrics files hash-bound to the evaluation completion record" % verified)
    else:
        st = "blocked"; d.append("the evaluation completion record carries no per-block metrics hashes (older record); files are hashed here only")
    checks["metrics_file_binding"] = {"status": st, "details": d}
    # dataset binding
    d = []
    ds = {k: v for k, v in {"evaluation": comp.get("dataset_summary_sha256"), "training_identity": identity.get("dataset_summary_sha256"),
                            "training_completion": tcomp.get("dataset_summary_sha256")}.items() if v}
    cs = {k: v for k, v in {"evaluation": comp.get("campaign_sha256"), "training_identity": identity.get("campaign_sha256")}.items() if v}
    if len(set(ds.values())) == 1 and len(set(cs.values())) <= 1 and ds:
        st = "passed"; d.append("dataset summary %s... and campaign %s... agree between evaluation and training" % (list(ds.values())[0][:12], (list(cs.values()) or ["?"])[0][:12]))
    elif not ds:
        st = "blocked"; d.append("no dataset_summary_sha256 recorded")
    else:
        st = "failed"; d.append("dataset or campaign identities disagree: %s %s" % (ds, cs))
    checks["dataset_binding"] = {"status": st, "details": d}
    # split provenance
    st, d = split_check(split, pde)
    checks["split_provenance"] = {"status": st, "details": d}
    # training completion
    d = []
    events = [ev.get("kind") if isinstance(ev, dict) else str(ev) for ev in (health.get("events") or [])]
    if train is None:
        st = "blocked"; d.append("training output not paired (evaluation stands alone)")
    else:
        st = "passed"
        if tcomp.get("status") != "completed":
            st = "flagged"; d.append("training status %r (%s)" % (tcomp.get("status") or health.get("status"), cohort))
        if events:
            st = "flagged"; d.append("health events: %s" % events)
        if health.get("test_records_touched") not in (0, None) or tcomp.get("test_records_touched") not in (0, None):
            st = "failed"; d.append("test records touched during training")
        if health.get("validation_records") not in (0, None):
            st = "failed"; d.append("validation records used")
        d.append("epochs %s/%s, stop %s, final train loss %s" % (cohort_evidence["epochs_completed"], cohort_evidence["planned_epochs"],
                                                                 cohort_evidence["stop_reason"], health.get("train_loss_final")))
    checks["training_completion"] = {"status": st, "details": d}
    # finiteness
    d = []
    nonfinite = [v for v in present if (blocks[v].get("metrics") or {}).get("nonfinite_rate") not in (None, 0, 0.0)]
    bad_rel = [v for v in present if not finite_number((blocks[v].get("metrics") or {}).get("relative_l2"))]
    bad_h = [v for v in present for h in (1, 2, 3) if ("rel_l2_h%d" % h) in (blocks[v].get("metrics") or {}) and not finite_number((blocks[v].get("metrics") or {}).get("rel_l2_h%d" % h))]
    if nonfinite or bad_rel or bad_h:
        st = "failed"; d.append("non-finite rate > 0 in %s; relative_l2 not a finite real scalar in %s; per-horizon values not finite in %s" % (nonfinite, bad_rel, sorted(set(bad_h))))
    else:
        st = "passed"; d.append("relative_l2 (and per-horizon values where present) is a finite real scalar in every block" + ("; recorded nonfinite_rate 0 in every block" if any("nonfinite_rate" in (blocks[v].get("metrics") or {}) for v in present) else "; no nonfinite_rate field in these records"))
    checks["finiteness"] = {"status": st, "details": d}
    checks["metric_recomputation"] = {"status": "blocked", "details": ["no prediction artifact was retained by the campaign runner; the metrics are the frozen evaluator's stored output and cannot be recomputed from arrays"]}
    core = [checks[k]["status"] for k in ("blocks_complete", "checkpoint_binding", "metrics_file_binding", "dataset_binding", "split_provenance", "finiteness", "training_completion")]
    if "failed" in core:
        overall = "inconsistent"
    elif "blocked" in core or train is None:
        overall = "incomplete"
    else:
        overall = "consistent"
    evaluator = comp.get("frozen_evaluator_sha256")
    scoring_version = ("legacy-campaign-evaluator:" + evaluator[:16]) if evaluator else "legacy-campaign-evaluator:unrecorded"
    out = {
        "identity": ident, "pde": pde, "method": method, "train_view": train_view, "task": task_of(pde),
        "attempt_identity": "eval-" + sel["files"]["completion.json"]["sha256"][:16],
        "evaluation_completion_sha256": sel["files"]["completion.json"]["sha256"],
        "evaluation_schema": comp.get("schema_version"), "evaluation_status": comp.get("status"),
        "scoring_version": scoring_version, "scoring_semantics": "campaign evaluator relative L2 (legacy family, docs/results_format.md); not pdeobs-strict-v1",
        "acceptance_policy": comp.get("acceptance_policy"), "policy_disposition": (comp.get("policy_evidence") or {}).get("disposition"),
        "training_cohort": cohort, "cohort_evidence": cohort_evidence,
        "actual_epochs": cohort_evidence["epochs_completed"], "stop_reason": cohort_evidence["stop_reason"],
        "checkpoint_id": comp.get("checkpoint_sha256"),
        "resolved_config_identity": ((train or {}).get("files") or {}).get("checkpoints/training_config.json", {}).get("sha256") if train else None,
        "training_config": public_training_config(rec) if train else None,
        "training_attempt_identity": ("train-" + (train["files"].get("completion.json") or train["files"].get("health.json") or train["files"].get("identity.json") or {}).get("sha256", "")[:16]) if train else None,
        "protocol_adapter_sha256": tcomp.get("protocol_adapter_sha256") or (rec.get("budget-protocol.json") or {}).get("adapter_sha256"),
        "registry_sha256": comp.get("registry_sha256") or identity.get("registry_sha256"),
        "campaign_sha256": comp.get("campaign_sha256"), "dataset_summary_sha256": comp.get("dataset_summary_sha256"),
        "repository_commit": comp.get("repository_commit"), "frozen_evaluator_sha256": evaluator,
        "split": {k: split.get(k) for k in ("schema_version", "algorithm", "seed", "records", "train_records", "test_records", "validation_records",
                                            "regime_counts", "test_sample_ids_sha256", "train_sample_ids_sha256", "macrodomain_identity")} if split else None,
        "health": {"status": health.get("status"), "events": events, "warnings": len(health.get("warnings") or []),
                   "train_loss_final": health.get("train_loss_final"), "train_loss_first": health.get("train_loss_first"),
                   "optimizer_steps": health.get("optimizer_steps")} if train else None,
        "blocks": {v: {"relative_l2": (blocks[v].get("metrics") or {}).get("relative_l2") if finite_number((blocks[v].get("metrics") or {}).get("relative_l2")) else "NONFINITE_OR_MISSING",
                       "per_horizon": {h: ((blocks[v].get("metrics") or {}).get("rel_l2_h%d" % h) if finite_number((blocks[v].get("metrics") or {}).get("rel_l2_h%d" % h)) else "NONFINITE_OR_MISSING") for h in (1, 2, 3) if ("rel_l2_h%d" % h) in (blocks[v].get("metrics") or {})},
                       "mse": (blocks[v].get("metrics") or {}).get("mse"), "mae": (blocks[v].get("metrics") or {}).get("mae"),
                       "nonfinite_rate": (blocks[v].get("metrics") or {}).get("nonfinite_rate"),
                       "samples": blocks[v].get("samples"), "metrics_sha256": (sel["files"].get("blocks/%s/metrics.json" % v) or {}).get("sha256"),
                       "config_hash": blocks[v].get("config_hash"), "mask_protocol": blocks[v].get("mask_protocol"),
                       "observation_ratio": blocks[v].get("observation_ratio"), "evaluation_config": blocks[v].get("evaluation_config")}
                   for v in present},
        "artifact_hashes": {"evaluation": sel["files"], "training": (train or {}).get("files") if train else None,
                            "checkpoint_files": (train or {}).get("checkpoints") if train else None},
        "checks": checks, "verification_status": overall,
    }
    return out


def cmd_index(args):
    paired = read_json(args.paired)
    if paired.get("schema_version") != PAIR_VERSION:
        sys.stderr.write("not a pair file\n")
        return 2
    if os.path.exists(args.out_dir):
        sys.stderr.write("refusing to write into the existing directory %s\n" % args.out_dir)
        return 2
    os.makedirs(args.out_dir)
    records, pending, failed_only = {}, [], []
    for ident, group in sorted(paired["identities"].items()):
        sel = group.get("selected")
        if not sel:
            continue
        records[ident] = build_record(ident, sel, sel.get("training"))
        records[ident]["selection_rule"] = group.get("selection_rule")
        records[ident]["alternative_evaluations"] = len(group.get("alternatives") or [])
    failures = {}
    for t in paired.get("unmatched_training", []):
        if t.get("identity"):
            failures.setdefault(t["identity"], []).append(t)
    for pde in PDES:
        for method in METHODS:
            for view in VIEWS:
                ident = "%s/%s/%s" % (pde, method, view)
                if ident in records:
                    continue
                attempts = failures.get(ident, [])
                pending.append({"identity": ident, "pde": pde, "method": method, "train_view": view, "task": task_of(pde),
                                "status": "NO_COMPLETED_EVALUATION", "value": MISSING, "failed_or_incomplete_attempts": len(attempts),
                                "attempt_events": sorted({e for a in attempts for e in (a.get("events") or [])}),
                                "latest_attempt_status": attempts[-1].get("status") if attempts else None})
    # long-format CSV
    cols = ["pde", "method", "train_view", "test_view", "task", "metric", "value", "identity_count", "training_cohort", "actual_epochs",
            "stop_reason", "attempt_identity", "training_attempt_identity", "checkpoint_id", "resolved_config_identity", "metrics_sha256",
            "test_identity_set_sha256", "scoring_version", "verification_status"]
    with io.open(os.path.join(args.out_dir, "results_long.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for ident, r in sorted(records.items()):
            for view in VIEWS:
                b = r["blocks"].get(view)
                base = [r["pde"], r["method"], r["train_view"], view, r["task"]]
                tail = [r["training_cohort"], r["actual_epochs"], r["stop_reason"], r["attempt_identity"], r["training_attempt_identity"],
                        r["checkpoint_id"], r["resolved_config_identity"]]
                if not b:
                    w.writerow(base + ["relative_l2", MISSING, None] + tail + [None, None, r["scoring_version"], r["verification_status"]])
                    continue
                ids_sha = (r["split"] or {}).get("test_sample_ids_sha256")
                w.writerow(base + ["relative_l2", b["relative_l2"], b["samples"]] + tail + [b["metrics_sha256"], ids_sha, r["scoring_version"], r["verification_status"]])
                for h, val in sorted((b.get("per_horizon") or {}).items()):
                    w.writerow(base + ["rel_l2_h%s" % h, val, b["samples"]] + tail + [b["metrics_sha256"], ids_sha, r["scoring_version"], r["verification_status"]])
        for p in pending:
            w.writerow([p["pde"], p["method"], p["train_view"], "", p["task"], "relative_l2", MISSING, None] + [None] * 7 + [None, None, None, p["status"]])
    # matched diagonal
    with io.open(os.path.join(args.out_dir, "matched_diagonal.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["pde", "method", "train_view", "task", "matched_relative_l2", "training_cohort", "actual_epochs", "stop_reason", "verification_status", "attempt_identity"])
        for pde in PDES:
            for method in METHODS:
                for view in VIEWS:
                    ident = "%s/%s/%s" % (pde, method, view)
                    r = records.get(ident)
                    if r and view in r["blocks"]:
                        w.writerow([pde, method, view, r["task"], r["blocks"][view]["relative_l2"], r["training_cohort"], r["actual_epochs"], r["stop_reason"], r["verification_status"], r["attempt_identity"]])
                    else:
                        w.writerow([pde, method, view, task_of(pde), MISSING, None, None, None, "NO_COMPLETED_EVALUATION", None])
    # cross-view matrices
    xdir = os.path.join(args.out_dir, "cross_view"); os.makedirs(xdir)
    for pde in PDES:
        for method in METHODS:
            with io.open(os.path.join(xdir, "%s__%s.csv" % (pde, method)), "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["train_view \\ test_view"] + VIEWS)
                for tv in VIEWS:
                    r = records.get("%s/%s/%s" % (pde, method, tv))
                    w.writerow([tv] + [(r["blocks"].get(v) or {}).get("relative_l2", MISSING) if r else MISSING for v in VIEWS])
    # public index + checks + summary
    cohorts, statuses = {}, {}
    for r in records.values():
        cohorts[r["training_cohort"]] = cohorts.get(r["training_cohort"], 0) + 1
        statuses[r["verification_status"]] = statuses.get(r["verification_status"], 0) + 1
    summary = {"schema_version": INDEX_VERSION, "identities_with_selected_evaluation": len(records), "planned_identities": len(PDES) * len(METHODS) * len(VIEWS),
               "identities_without_completed_evaluation": len(pending), "blocks_indexed": sum(len(r["blocks"]) for r in records.values()),
               "by_training_cohort": cohorts, "by_verification_status": statuses,
               "selection_rules": sorted({r["selection_rule"] for r in records.values()}),
               "scoring": "campaign evaluator relative L2 (legacy family); no strict-v1 blocks and no prediction tensors exist for these runs",
               "not_established": ["numerical-reference accuracy of the evaluated targets", "external reproduction of any run",
                                   "recomputation of any metric from arrays", "checkpoint file re-hash unless the scan ran with --hash-checkpoints"]}
    write_json(os.path.join(args.out_dir, "index.json"), {"schema_version": INDEX_VERSION, "summary": summary, "records": records, "pending": pending})
    write_json(os.path.join(args.out_dir, "checks.json"), {"schema_version": INDEX_VERSION, "checks": {k: {"verification_status": v["verification_status"], "checks": v["checks"]} for k, v in records.items()}})
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    write_json(os.path.join(args.out_dir, "pending.json"), pending)
    sys.stderr.write("index: %s\n" % json.dumps({k: summary[k] for k in ("identities_with_selected_evaluation", "identities_without_completed_evaluation", "by_training_cohort", "by_verification_status")}))
    return 0


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python tools/archived_results.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    s = sub.add_parser("scan", help="walk archived roots on the machine that holds them; JSON records parsed and hashed, no tensors")
    s.add_argument("--training-root"); s.add_argument("--evaluation-root"); s.add_argument("--label", help="host or archive label recorded in the scan")
    s.add_argument("--out", required=True, help="NEW file, or - for stdout"); s.add_argument("--hash-checkpoints", action="store_true", help="also SHA-256 every checkpoints/*.pt (slow)")
    p = sub.add_parser("pair", help="join scans: evaluation to training by checkpoint identity, one selected attempt per identity")
    p.add_argument("--scan", nargs="+", required=True); p.add_argument("--selection", help="JSON {identity: {evaluation_completion_sha256 | evaluation_rel | checkpoint_sha256}}")
    p.add_argument("--out", required=True, help="NEW file")
    i = sub.add_parser("index", help="write the public index, checks, long-format CSV and pivots into a NEW directory")
    i.add_argument("--paired", required=True); i.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "pair":
        return cmd_pair(args)
    if args.command == "index":
        return cmd_index(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

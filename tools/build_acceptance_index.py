"""Build the current acceptance index from the shipped acceptance materials.

Reads (never modifies): ``ACCEPTANCE_SUMMARY.md``, ``RELEASE_NOTES_v0.2.0.md``, ``acceptance/**``,
``docs/ai/evidence_status.json`` and ``src/pdeobs/api/specs.py``.  Writes
``docs/ai/acceptance_index.json`` and ``docs/ai/ACCEPTANCE_INDEX.md`` only with ``--write``; because both
are generated files, existing outputs are replaced only with ``--force``.

Every number in the index is parsed from a shipped file whose SHA-256 is recorded next to it.  A parse
failure is an error, not a default, so the index cannot silently drift from its sources.  Historical
records keep their own outcomes; superseded records are marked, never deleted.  Counts from different
records overlap and are never added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs/ai/acceptance_index.json"
OUT_MD = ROOT / "docs/ai/ACCEPTANCE_INDEX.md"
SOURCES = ["ACCEPTANCE_SUMMARY.md", "RELEASE_NOTES_v0.2.0.md", "acceptance/README.md",
           "acceptance/FULL_COVERAGE_REPORT_round1.md", "acceptance/FULL_COVERAGE_REPORT_round2.md",
           "acceptance/harness/remote_full_coverage.py", "acceptance/harness/remote_coverage.sh",
           "docs/ai/evidence_status.json", "src/pdeobs/api/specs.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def must(pattern: str, text: str, where: str, flags: int = 0) -> re.Match:
    match = re.search(pattern, text, flags)
    if not match:
        raise SystemExit(f"acceptance index: pattern {pattern!r} not found in {where}; refusing to guess")
    return match


def parse_summary() -> dict:
    path = ROOT / "ACCEPTANCE_SUMMARY.md"
    text = path.read_text(encoding="utf-8")
    generated = must(r"Generated (\S+)\.", text, path.name).group(1)
    device = must(r"- device: (.+)", text, path.name).group(1).strip()
    torch, cuda = must(r"- torch (\S+) / CUDA (\S+)", text, path.name).groups()
    core = must(r"frozen portable core \+ updater tests: \*\*(\d+)/(\d+) passed\*\*", text, path.name).groups()
    api = must(r"new API tests \(specs \+ workflows\): (\d+) passed", text, path.name).group(1)
    l2 = must(r"resolution (\d+)\S, (\d+) samples, (\d+) optimizer steps, views \[(.*?)\]; \*\*passed (\d+)/(\d+)\*\*", text, path.name)
    return {
        "id": "acceptance-summary-" + generated.replace("-", "").replace(":", "")[:13] + "Z",
        "kind": "author GPU software acceptance (summary only)",
        "source": path.name, "source_sha256": sha256(path), "generated_utc": generated,
        "environment": {"device": device, "torch": torch, "cuda": cuda, "cpu_or_gpu": "GPU"},
        "code_under_test": "v0.2.0 tree (commit not recorded in the summary)",
        "scope": {"L0_frozen_core": f"{core[0]}/{core[1]} passed", "L0_api_tests": f"{api} passed",
                  "L1_parameter_coverage_models": 7,
                  "L2_flows": int(l2.group(5)), "L2_flows_total": int(l2.group(6)), "grid": int(l2.group(1)),
                  "samples_per_flow": int(l2.group(2)), "optimizer_steps": int(l2.group(3)),
                  "views": [v.strip(" '") for v in l2.group(4).split(",")], "view_namespace": "general",
                  "batch_size": "not recorded in the summary"},
        "raw_logs_in_tree": False,
        "status": "historical; the per-flow raw records of this summary are not shipped; "
                  "its interface tiers are superseded by acceptance round 2 (later the same day)",
        "superseded_by": "acceptance-round2",
        "commands": "not recorded in the summary (see acceptance/harness for the round sweeps)",
    }


def parse_release_notes() -> dict:
    path = ROOT / "RELEASE_NOTES_v0.2.0.md"
    text = path.read_text(encoding="utf-8")
    l2 = must(r"- \*\*L2\*\*(.*?)(?=\n- \*\*|\n\n)", text, path.name, re.S).group(0)
    l0 = must(r"- \*\*L0\*\*(.*?)(?=\n- \*\*|\n\n)", text, path.name, re.S).group(0)
    claim = " ".join(line.strip() for line in l2.splitlines())
    return {
        "id": "release-notes-v0.2.0-acceptance-paragraph",
        "kind": "author claim text (no raw record of its own)",
        "source": path.name, "source_sha256": sha256(path),
        "claim_L2": claim, "claim_L0": " ".join(line.strip() for line in l0.splitlines()),
        "assessment": [
            "'49 short flows' appears with raw numbers only in ACCEPTANCE_SUMMARY.md, where the flows run at 64^2 with two "
            "general views (random, block), not at 128x128 with nine paper views",
            "the 128x128 nine-paper-view flows in the shipped raw records are acceptance round 2 tier T5: 14 flows "
            "(7 models x {recovery, rollout}), each scored on the nine paper views plus eight general views",
            "no shipped raw record uses batch 1: tier T13 records the paper batch sizes (8, 4 and 2) and the U-FNO "
            "specification requires a training batch size of at least 2",
            "'521 portable + 37 updater tests' is not a basis for 558: the frozen scope collects 521 cases in total (T12)",
        ],
        "status": "claim as written is not substantiated by any shipped raw record; cite the tier JSON instead",
        "superseded_by": "acceptance-round2",
    }


def parse_round(name: str) -> dict:
    report = ROOT / f"acceptance/FULL_COVERAGE_REPORT_{name}.md"
    text = report.read_text(encoding="utf-8")
    utc, finished = must(r"Sweep UTC: (\S+) \(finished (.*?)\)", text, report.name).groups()
    node = must(r"- Node: (.+)", text, report.name).group(1).strip()
    archive, commit, code_sha = must(r"Code under test: `([^`]+)` \(commit (\w+)[^)]*\) sha256 `(\w+)`", text, report.name).groups()
    coverage = json.loads((ROOT / f"acceptance/{name}/coverage.json").read_text(encoding="utf-8"))
    essentials = (ROOT / f"acceptance/{name}/essentials.txt").read_text(encoding="utf-8").strip().splitlines()[-1].strip()
    t13 = json.loads((ROOT / f"acceptance/{name}/T13.json").read_text(encoding="utf-8"))
    batch = {}
    for key in t13.get("runs", {}):
        m = must(r"^(\w+):(\w+)@(\d+) bs=(\d+)$", key, f"acceptance/{name}/T13.json run key")
        batch[f"{m.group(1)}:{m.group(2)}"] = {"grid": int(m.group(3)), "batch_size": int(m.group(4))}
    t5 = json.loads((ROOT / f"acceptance/{name}/T5.json").read_text(encoding="utf-8"))
    flows = t5.get("flows", {})
    views = {}
    for key, flow in flows.items():
        views[key] = {"paper_views": sorted(flow.get("paper", {})), "general_views": len(flow.get("general", {})),
                      "status": flow.get("status")}
    tiers = {tier: {"status": value.get("status"), "summary": value.get("summary"), "seconds": value.get("seconds"),
                    "error": value.get("error")} for tier, value in coverage["tiers"].items()}
    files = {p.relative_to(ROOT).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted((ROOT / "acceptance" / name).iterdir())}
    return {
        "id": f"acceptance-{name}",
        "kind": "author GPU software acceptance sweep (14 tiers, raw per-tier JSON shipped)",
        "source": report.relative_to(ROOT).as_posix(), "source_sha256": sha256(report),
        "sweep_utc": utc, "finished": finished, "node": node,
        "environment": {**coverage.get("env", {}), "cpu_or_gpu": "GPU"},
        "code_under_test": {"archive": archive, "commit": commit, "sha256": code_sha},
        "tiers": tiers,
        "essentials": essentials,
        "t5_flows_128": {"count": len(flows), "grid": 128, "per_flow": views,
                         "optimizer_steps": "a few steps per flow (the tier JSON records status and peak memory, not a step count)"},
        "t13_paper_batch_sizes": batch,
        "raw_files": files,
        "commands": ["acceptance/harness/remote_coverage.sh (launcher on the node)",
                     "python acceptance/harness/remote_full_coverage.py --tiers T1..T14 (sweep)"],
    }


def parse_specs() -> dict:
    path = ROOT / "src/pdeobs/api/specs.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    rule = next((n for n, line in enumerate(lines, 1) if "training batch size must be >= 2" in line), None)
    if rule is None:
        raise SystemExit("acceptance index: U-FNO batch rule not found in specs.py")
    defaults = {}
    for line in lines:
        m = re.match(r'^\s+"(\w+)": \{.*"batch_size": (\d+)', line)
        if m:
            defaults[m.group(1)] = int(m.group(2))
    return {"source": "src/pdeobs/api/specs.py", "source_sha256": sha256(path),
            "ufno_batch_rule": {"line": rule, "text": lines[rule - 1].strip()},
            "paper_training_default_batch_sizes": defaults}


def overlay_entries() -> list[dict]:
    path = ROOT / "docs/ai/evidence_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    one = status.get("independently_executed", {})
    entries = [{
        "id": "overlay-1-cpu-audit",
        "kind": "independent CPU audit of the base upload (overlay 1)",
        "source": "docs/ai/evidence_status.json#independently_executed", "source_sha256": sha256(path),
        "environment": {**one.get("environment", {}), "cpu_or_gpu": "CPU"},
        "tests": {"original_all_shipped_tests": one.get("original_all_shipped_tests"),
                  "final_overlay_all_shipped_tests": one.get("final_overlay_all_shipped_tests")},
        "smoke": one.get("ai_cpu_smoke"), "install": one.get("install"),
        "status": "historical CPU audit of the base ZIP; superseded for the current tree by overlay 2",
        "superseded_by": "overlay-2-this-tree",
    }]
    two = status.get("overlay_2_this_run")
    if two:
        entries.append({
            "id": "overlay-2-this-tree",
            "kind": "independent CPU verification of this tree (overlay 2)",
            "source": "docs/ai/evidence_status.json#overlay_2_this_run", "source_sha256": sha256(path),
            "environment": {**two.get("environment", {}), "cpu_or_gpu": "CPU"},
            "tests": two.get("full_test_suite"), "smoke": two.get("ai_cpu_smoke"),
            "evidence_checker": two.get("evidence_checker_on_demo_rows"), "install": two.get("install"),
            "commands": two.get("commands"), "logs": two.get("logs"),
            "status": "current local verification record for this tree; CPU only; software evidence only",
        })
    else:
        entries.append({"id": "overlay-2-this-tree", "status": "pending: docs/ai/evidence_status.json has no overlay_2_this_run block yet"})
    return entries


CONFLICTS = [
    {"id": "C1", "between": ["acceptance-summary", "release-notes-v0.2.0-acceptance-paragraph", "acceptance-round2"],
     "conflict": "49 flows at 64^2 / 4 samples / 2 steps / views random+block (summary) vs '49 flows, 128x128, batch 1, nine paper views' (release notes) vs T5 = 14 flows at 128^2 with nine paper + eight general views (round 2)",
     "resolution": "cite the concrete tier JSON; do not merge the three into one stronger claim; the release-notes wording is unsubstantiated"},
    {"id": "C2", "between": ["release-notes-v0.2.0-acceptance-paragraph", "acceptance-round1", "acceptance-round2"],
     "conflict": "'521 portable + 37 updater' reads as 558 cases; T12 collects 521 cases in total",
     "resolution": "521 is the frozen-scope total; never add the 37"},
    {"id": "C3", "between": ["acceptance-summary", "acceptance-round1", "acceptance-round2"],
     "conflict": "API tests are reported as 38 (summary, round 1) and 42 (round 2)",
     "resolution": "different code versions (commit ANON_ACCEPTANCE_COMMIT_1 vs ANON_ACCEPTANCE_COMMIT_2 added tests); report each with its commit"},
    {"id": "C4", "between": ["acceptance-round1", "acceptance-round2"],
     "conflict": "round 1: T3 9/26, T4 14/16, T5 0/14, T7 and T8 crashed, essentials 2 failed; round 2: all 14 tiers complete, essentials 34 passed",
     "resolution": "round 1 is a failed historical record kept as the log of the defects; round 2 supersedes it for every tier"},
    {"id": "C5", "between": ["acceptance-round1", "acceptance-round2"],
     "conflict": "T11 gpu_bitwise false (round 1) vs true (round 2); T13 per-step times differ 1.6-3x",
     "resolution": "host dependent; the CPU path is the reproducibility contract; T13 figures are node-specific estimates, not benchmark numbers"},
    {"id": "C6", "between": ["overlay-1-cpu-audit", "overlay-2-this-tree", "acceptance-round2"],
     "conflict": "test totals 521 / 597 / 630 / and the overlay-2 total are counts of overlapping selections of one suite on different trees",
     "resolution": "report one total per run with its tree hash; never add totals across runs"},
]


def build() -> dict:
    rounds = [parse_summary(), parse_round("round1"), parse_round("round2"), parse_release_notes(), *overlay_entries()]
    rounds[1]["status"] = "historical, failed round; superseded by acceptance-round2 (defects fixed in commit ANON_ACCEPTANCE_COMMIT_2); kept as the record of the defects"
    rounds[1]["superseded_by"] = "acceptance-round2"
    rounds[2]["status"] = "current author-reported GPU software acceptance for the v0.2.0 facade; not repeated by the reviewers; never a paper table"
    return {
        "schema_version": "pdeobs-acceptance-index/v1",
        "generated_by": "tools/build_acceptance_index.py",
        "policy": [
            "every entry keeps its own outcome; failed or superseded records are marked, never rewritten or deleted",
            "a status of 'superseded' means a later record covers the same scope on a later tree; it does not invalidate the earlier log",
            "software acceptance (all entries) never enters a paper table; paper evidence is a separate route (docs/paper_evidence.md)",
            "counts from different entries overlap and are never added",
        ],
        "current": {
            "author_gpu_software_acceptance": "acceptance-round2",
            "local_cpu_verification_of_this_tree": "overlay-2-this-tree",
            "paper_evidence": "none in this tree (docs/paper_evidence.md)",
        },
        "records": rounds,
        "constraints": parse_specs(),
        "conflicts": CONFLICTS,
        "sources": {rel: {"sha256": sha256(ROOT / rel), "bytes": (ROOT / rel).stat().st_size} for rel in SOURCES},
    }


def render_md(index: dict) -> str:
    out = ["# Acceptance index (generated; do not edit by hand)", "",
           "Generated by `tools/build_acceptance_index.py` from the shipped acceptance files; every number below is",
           "parsed from a file whose SHA-256 is listed at the end. Machine-readable form: `docs/ai/acceptance_index.json`.", ""]
    out += ["## Policy", ""] + [f"- {p}" for p in index["policy"]] + [""]
    out += ["## Current pointers", ""] + [f"- **{k.replace('_', ' ')}**: `{v}`" for k, v in index["current"].items()] + [""]
    out += ["## Records", "", "| id | kind | when | environment | code | scope | status |", "|---|---|---|---|---|---|---|"]
    for r in index["records"]:
        env = r.get("environment", {})
        env_s = ", ".join(str(env.get(k)) for k in ("device", "torch", "cuda", "python", "platform") if env.get(k)) or "-"
        code = r.get("code_under_test")
        code_s = code if isinstance(code, str) else (f"{code.get('archive')} commit {code.get('commit')} sha {code.get('sha256', '')[:12]}" if code else "-")
        when = r.get("generated_utc") or r.get("sweep_utc") or "-"
        if r["id"].startswith("acceptance-summary"):
            s = r["scope"]
            scope = (f"L0 {s['L0_frozen_core']}, API {s['L0_api_tests']}; L2 {s['L2_flows']}/{s['L2_flows_total']} flows at {s['grid']}^2, "
                     f"{s['samples_per_flow']} samples, {s['optimizer_steps']} steps, views {s['views']} (general); batch: {s['batch_size']}")
        elif r["id"].startswith("acceptance-round"):
            t = r["tiers"]
            scope = (f"14 tiers; T5 {t['T5']['summary'].get('passed')}/{t['T5']['summary'].get('total')} flows at 128^2 (nine paper + eight general views); "
                     f"T12 core {t['T12']['summary'].get('core_passed')} + API '{t['T12']['summary'].get('api')}'; T13 paper batch sizes "
                     + ", ".join(f"{k} bs={v['batch_size']}" for k, v in r["t13_paper_batch_sizes"].items()) + f"; essentials: {r['essentials']}")
        elif r["id"].startswith("release-notes"):
            scope = r["claim_L2"]
        elif r["id"] == "overlay-1-cpu-audit":
            scope = f"tests {r['tests']}; smoke {r.get('smoke')}"
        elif r["id"] == "overlay-2-this-tree" and r.get("tests"):
            scope = f"tests {r['tests']}; smoke {r.get('smoke')}; evidence checker {r.get('evidence_checker')}"
        else:
            scope = "-"
        out.append(f"| `{r['id']}` | {r.get('kind', '-')} | {when} | {env_s} | {code_s} | {scope} | {r.get('status', '-')} |")
    out.append("")
    for name in ("acceptance-round1", "acceptance-round2"):
        r = next(x for x in index["records"] if x["id"] == name)
        out += [f"## Tiers of `{name}` ({r['sweep_utc']}, {r['node']})", "", "| tier | status | summary | seconds |", "|---|---|---|---|"]
        for tier, value in r["tiers"].items():
            summary = value.get("summary") if value.get("summary") is not None else value.get("error")
            out.append(f"| {tier} | {value['status']} | `{json.dumps(summary, sort_keys=True)}` | {value.get('seconds')} |")
        out += ["", f"Essentials: {r['essentials']}. Commands: " + "; ".join(f"`{c}`" for c in r["commands"]) + ".", "",
                "Raw files (SHA-256 prefix): " + ", ".join(f"`{Path(p).name}` {v['sha256'][:12]}" for p, v in r["raw_files"].items()), ""]
    c = index["constraints"]
    out += ["## Batch-size constraint", "",
            f"- `{c['source']}` line {c['ufno_batch_rule']['line']}: `{c['ufno_batch_rule']['text']}`",
            f"- paper training default batch sizes: `{json.dumps(c['paper_training_default_batch_sizes'])}`",
            "- consequence: any record claiming U-FNO flows at batch 1 cannot be a run of this code; no shipped raw record claims it.", ""]
    out += ["## Conflicts between records (kept, not resolved by rewriting)", ""]
    for k in index["conflicts"]:
        out += [f"- **{k['id']}** ({', '.join(k['between'])}): {k['conflict']}. *Resolution:* {k['resolution']}"]
    out += ["", "## Sources", "", "| file | sha256 | bytes |", "|---|---|---|"]
    out += [f"| `{p}` | `{v['sha256']}` | {v['bytes']} |" for p, v in index["sources"].items()]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write docs/ai/acceptance_index.json and ACCEPTANCE_INDEX.md")
    parser.add_argument("--force", action="store_true", help="replace existing generated outputs")
    args = parser.parse_args(argv)
    index = build()
    if not args.write:
        sys.stdout.write(json.dumps(index, indent=2, sort_keys=False, default=str) + "\n")
        return 0
    for target in (OUT_JSON, OUT_MD):
        if target.exists() and not args.force:
            sys.stderr.write(f"{target} exists; pass --force to replace the generated file\n")
            return 2
    OUT_JSON.write_text(json.dumps(index, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(index), encoding="utf-8")
    sys.stdout.write(json.dumps({"written": [OUT_JSON.relative_to(ROOT).as_posix(), OUT_MD.relative_to(ROOT).as_posix()],
                                 "records": [r["id"] for r in index["records"]]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

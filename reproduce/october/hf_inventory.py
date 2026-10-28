"""Inventory of the public Hugging Face deposits at pinned revisions, reconciled with the manifests they contain.

    python reproduce/october/hf_inventory.py --out results/public_deposits/hf_inventory [--cache-dir <dir>] [--offline]
                                             [--dataset-revision <sha>] [--model-revision <sha>] [--code-root <checkout>]

Without credentials, for the dataset and the model repository ru1ch3n/PDE-OBS:
1. resolve the revision (default: the current main) to its commit;
2. list every file at that commit through the paginated tree API (path, size, LFS SHA-256 or git blob id) and write
   <out>/inventory_<kind>_<commit7>.csv.gz sorted by path (the SHA-256 of the CSV text goes into <out>/summary.json);
3. fetch every manifest the repository holds at that commit and check each entry with the rules of its schema (below):
   the file must exist at the path the schema defines and carry the recorded SHA-256 (an LFS object is compared through
   the digest the tree API reports, a regular file of at most 4 MB is fetched and hashed), its recorded size must equal
   the listed size, aggregate fields are recomputed (file counts, bytes, file types and shard digests of the V2 views),
   entries a manifest declares as not published must be absent, run ids a part declares as held by another part must be
   found there, and entries that describe files of the benchmark code repository are checked against the checkout;
4. write <out>/reconciliation.json (per manifest: present, verified, missing, digest or size mismatches, aggregates,
   declared absences; then the files no manifest covers, by class and top-level directory) and
   <out>/release_bindings.json (repositories, commits, file counts, bytes, the SHA-256 of every manifest and of both
   inventories, the code-repository files the results manifests describe).

Manifest schemas and where their entries live:
  release_manifest*.json  (schema_version 1)             files[] {path, sha256, size}                 data/<path>
  scrub-manifest.json     (dataset)                      shards {path: scrubbed_sha256, bytes}        data/<path>; summary_files -> <name>
  data_manifest.part-N.json (october2026-data-manifest/1) shards[] {path, sha256, bytes}; views {}     october2026/<path>; aggregates per view
  manifest.part-X.json    (pdeobs-october-results/v1)    export_files {name: sha256}                  <manifest dir>/<name>
                                                          committed_files {name: sha256, rows}         results/revision_october/<name> of the code repository
  export_manifest.json    (pdeobs-october-export/v1)     files {name: sha256}, published_files        <published_under>/<name>; not_published -> absent
  models_manifest.cluster-X.json (anonymized-release/v3) rows[].files {relpath: sha256, bytes}        models/<host_label>/<release_path>/<relpath>
  models_manifest.part-N.json (october2026-models-manifest/1) models[] {cohort, run_id, weights_sha256, files[]}
                                                                                                       october2026/models/<cohort>/<run_id>/<file>
  retrains/manifest.json  (october2026-retrains-manifest/1) models[] {name, weights_sha256, files[]}  october2026/retrains/<name>/<file>; not_published -> absent
  scrub-manifest.json     (model)                        rewritten_files[] (paths; digests are in the cluster manifests)

Standard library only. The listing is metadata: no data shard and no checkpoint is downloaded. The regular files that
are fetched to verify a digest are the small JSON / CSV / YAML sidecars and training records (at most 4 MB each); with
--cache-dir their digests, the manifests and the listing are kept so that a later run (--offline) needs no network.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HUB = "https://huggingface.co"
UA = {"User-Agent": "pdeobs-public-inventory/2"}
MANIFEST_RE = re.compile(r"(^|/)(release_manifest(_full)?\.json|scrub-manifest\.json|data_manifest\.part-\d+\.json|models_manifest[^/]*\.json|"
                         r"manifest\.part-[A-Z]\.json|export_manifest\.json|retrains/manifest\.json)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOC_RE = re.compile(r"(^|/)(README\.md|LICENSE[^/]*|\.gitattributes|[^/]+\.md)$")
SIDECAR_RE = re.compile(r"\.(sha256|manifest\.json|metadata\.json|metadata\.csv|quality\.json)$")
SMALL = 4 << 20
PACE = 0.36
LIST_LIMIT = 50


class Hub:
    """Paced, retrying access to the Hub (one request start every `pace` seconds across threads, 60 s pause on HTTP 429)
    with an optional on-disk cache: <cache_dir>/<kind>_<commit7>/{info.json, tree.json, digests.json, manifests/<path>}."""

    def __init__(self, cache_dir: str | None = None, offline: bool = False, pace: float = PACE, threads: int = 3):
        self.cache_dir, self.offline, self.pace, self.threads = cache_dir, offline, pace, max(1, threads)
        self.lock = threading.Lock()
        self.next_slot = 0.0
        self.requests = self.retries_429 = self.other_retries = 0

    def _wait(self) -> None:
        with self.lock:
            t = max(time.monotonic(), self.next_slot)
            self.next_slot = t + self.pace
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def get(self, url: str, timeout: int = 300, retries: int = 10) -> tuple[bytes, dict]:
        if self.offline:
            raise RuntimeError("offline run, no request made: " + url)
        for attempt in range(retries):
            self._wait()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                    with self.lock:
                        self.requests += 1
                    return r.read(), {k.lower(): v for k, v in r.headers.items()}
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    with self.lock:
                        self.retries_429 += 1
                        self.next_slot = max(self.next_slot, time.monotonic() + 60)
                    continue
                if e.code in (500, 502, 503, 504) and attempt < retries - 1:
                    with self.lock:
                        self.other_retries += 1
                    time.sleep(20)
                    continue
                raise
            except (urllib.error.URLError, OSError):
                if attempt == retries - 1:
                    raise
                with self.lock:
                    self.other_retries += 1
                time.sleep(15)
        raise RuntimeError("too many retries: " + url)

    def cache_path(self, kind: str, commit: str, *parts: str) -> str | None:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{kind}_{commit[:7]}", *parts)

    def cached(self, kind: str, commit: str, *parts: str) -> bytes | None:
        p = self.cache_path(kind, commit, *parts)
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        return None

    def store(self, kind: str, commit: str, data: bytes, *parts: str) -> None:
        p = self.cache_path(kind, commit, *parts)
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)


def api_kind(kind: str) -> str:
    return "datasets" if kind == "dataset" else "models"


def resolve_revision(kind: str, repo: str, revision: str, get) -> dict:
    b, _ = get(f"{HUB}/api/{api_kind(kind)}/{repo}/revision/{urllib.parse.quote(revision, safe='')}")
    d = json.loads(b)
    return {"commit": d["sha"], "last_modified": d.get("lastModified"), "private": d.get("private"), "gated": d.get("gated")}


def next_link(headers: dict) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link") or "")
    return m.group(1) if m else None


def list_tree(kind: str, repo: str, commit: str, get) -> tuple[list[dict], int]:
    """Every file of the repository at the commit: [{path, size, lfs, sha256 (LFS objects), blob (git id otherwise)}],
    and the number of pages read."""
    url = f"{HUB}/api/{api_kind(kind)}/{repo}/tree/{commit}?recursive=true&expand=true"
    files, pages = [], 0
    while url:
        b, h = get(url)
        pages += 1
        for e in json.loads(b):
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            files.append({"path": e["path"], "size": int(e.get("size") or 0), "lfs": bool(lfs),
                          "sha256": (lfs.get("sha256") or lfs.get("oid")) if lfs else None, "blob": None if lfs else e.get("oid")})
        url = next_link(h)
    files.sort(key=lambda f: f["path"])
    return files, pages


def inventory_csv(files: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["path", "size", "lfs", "sha256", "blob"])
    for f in files:
        w.writerow([f["path"], f["size"], int(f["lfs"]), f["sha256"] or "", f["blob"] or ""])
    return buf.getvalue().encode("utf-8")


def write_inventory(path: str, files: list[dict]) -> dict:
    data = inventory_csv(files)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(data)
    with open(path, "rb") as f:
        gz_sha = hashlib.sha256(f.read()).hexdigest()
    return {"file": os.path.basename(path), "csv_sha256": hashlib.sha256(data).hexdigest(), "gz_sha256": gz_sha, "rows": len(files)}


def detect_schema(doc, manifest_path: str) -> str:
    if not isinstance(doc, dict):
        return "unknown"
    base = manifest_path.rsplit("/", 1)[-1]
    fmt, schema = doc.get("format"), doc.get("schema")
    if schema == "pdeobs-october-export/v1" or (base == "export_manifest.json" and "files" in doc):
        return "export"
    if schema == "pdeobs-october-results/v1" or ("committed_files" in doc and "export_files" in doc):
        return "results"
    if fmt == "pdeobs-october2026-data-manifest/1" or ("views" in doc and "shards" in doc):
        return "data"
    if fmt == "pdeobs-october2026-models-manifest/1":
        return "october_models"
    if fmt == "pdeobs-october2026-retrains-manifest/1" or manifest_path.endswith("retrains/manifest.json"):
        return "retrains"
    if doc.get("schema_version") == "pdeobs-anonymized-release/v3" or ("rows" in doc and "host_label" in doc):
        return "cluster_models"
    if base == "scrub-manifest.json":
        return "scrub_dataset" if isinstance(doc.get("shards"), dict) else "scrub_model"
    if isinstance(doc.get("files"), list) and doc.get("schema_version") == 1:
        return "release"
    return "unknown"


def manifest_entries(doc, manifest_path: str) -> dict:
    """The checks a manifest asks for, resolved to repository paths by its schema:
    files [{path, sha256 | None, bytes | None, label}], absent [{path (a file, or a directory prefix ending in /), label,
    declared_by}], code_files [{name, sha256, rows}], views [{view, dir, files, bytes, shards, types, shard_digest,
    shard_lines}], run_ids, not_in_this_part, notes."""
    schema = detect_schema(doc, manifest_path)
    d = manifest_path.rsplit("/", 1)[0] + "/" if "/" in manifest_path else ""
    ent: dict = {"schema": schema, "files": [], "absent": [], "code_files": [], "views": [], "run_ids": [], "not_in_this_part": [], "notes": []}

    def add(path: str, sha: str | None = None, nbytes=None, label: str | None = None) -> None:
        ent["files"].append({"path": path, "sha256": sha if isinstance(sha, str) and SHA256_RE.match(sha) else None,
                             "bytes": nbytes if isinstance(nbytes, int) and not isinstance(nbytes, bool) else None, "label": label or path})

    if schema == "release":
        for e in doc.get("files") or []:
            add("data/" + e["path"], e.get("sha256"), e.get("size"))
            if e.get("url") and not str(e["url"]).endswith("/data/" + e["path"]):
                ent["notes"].append("url does not end with the path: " + e["path"])
    elif schema == "scrub_dataset":
        for p, v in (doc.get("shards") or {}).items():
            add("data/" + p, v.get("scrubbed_sha256"), v.get("bytes"))
        for name, v in (doc.get("summary_files") or {}).items():
            add(name, v.get("scrubbed_sha256"), v.get("scrubbed_bytes"))
    elif schema == "data":
        shards = doc.get("shards") or []
        for s in shards:
            add("october2026/" + s["path"], s.get("sha256"), s.get("bytes"))
        for view, v in (doc.get("views") or {}).items():
            ent["views"].append({"view": view, "dir": "october2026/" + view + "/", "files": v.get("files"), "bytes": v.get("bytes"),
                                 "shards": v.get("shards"), "types": v.get("types"), "shard_digest": v.get("shard_digest"),
                                 "shard_lines": [f"{s['path']} {s.get('sha256')}" for s in shards if s["path"].startswith(view + "/")]})
    elif schema == "results":
        for name, v in (doc.get("export_files") or {}).items():
            add(d + name, v.get("sha256"), v.get("bytes"), label=name)
        for name, v in (doc.get("committed_files") or {}).items():
            ent["code_files"].append({"name": name, "sha256": v.get("sha256"), "rows": v.get("rows")})
        for x in doc.get("excluded") or []:
            ent["notes"].append("excluded: " + str(x))
    elif schema == "export":
        files = doc.get("files") or {}
        published = doc.get("published_files")
        pub = set(published) if published is not None else set(files)
        not_pub = set(doc.get("not_published") or [])
        under = doc.get("published_under")
        base = (under.rstrip("/") + "/") if under else d
        if under and base != d:
            ent["notes"].append(f"published_under {under} differs from the manifest directory {d}")
        for name, v in files.items():
            if name in not_pub:
                ent["absent"].append({"path": base + name, "label": name, "declared_by": manifest_path, "sha256": v.get("sha256")})
            elif name in pub:
                add(base + name, v.get("sha256"), v.get("bytes"), label=name)
            else:
                ent["notes"].append("neither published nor declared not published: " + name)
        for name in sorted(pub - set(files)):
            ent["notes"].append("published_files names a file without a digest entry: " + name)
    elif schema == "cluster_models":
        host = doc.get("host_label") or ""
        for row in doc.get("rows") or []:
            base = f"models/{host}/{row.get('release_path')}/"
            fl = row.get("files") or {}
            for rel, v in fl.items():
                add(base + rel, v.get("sha256"), v.get("bytes"))
            ck = (fl.get("checkpoints/last.pt") or {}).get("sha256")
            if row.get("checkpoint_sha256") and ck != row.get("checkpoint_sha256"):
                ent["notes"].append("checkpoint_sha256 differs from files[checkpoints/last.pt].sha256: " + str(row.get("release_path")))
    elif schema == "october_models":
        for m in doc.get("models") or []:
            base = f"october2026/models/{m.get('cohort')}/{m['run_id']}/"
            ent["run_ids"].append(m["run_id"])
            for f in m.get("files") or ["weights.pt"]:
                if f == "weights.pt":
                    add(base + f, m.get("weights_sha256"), m.get("weights_bytes"))
                else:
                    add(base + f)
        ent["not_in_this_part"] = [str(x) for x in doc.get("not_in_this_part") or []]
        if doc.get("n_models") is not None and doc["n_models"] != len(doc.get("models") or []):
            ent["notes"].append("n_models differs from the number of model rows")
    elif schema == "retrains":
        for m in doc.get("models") or []:
            base = f"october2026/retrains/{m['name']}/"
            for f in m.get("files") or ["weights.pt"]:
                if f == "weights.pt":
                    add(base + f, m.get("weights_sha256"), m.get("weights_bytes"))
                else:
                    add(base + f)
        for x in doc.get("not_published") or []:
            name = x.get("name") if isinstance(x, dict) else str(x)
            ent["absent"].append({"path": f"october2026/retrains/{name}/", "label": name, "declared_by": manifest_path,
                                  "status": x.get("status") if isinstance(x, dict) else None})
    elif schema == "scrub_model":
        for p in doc.get("rewritten_files") or []:
            add(str(p))
    else:
        ent["notes"].append("unknown schema: entries are dicts with a sha256 and a path-like field, resolved at the path and below the manifest directory")

        def walk(node, parent_key=None):
            if isinstance(node, dict):
                sha = next((v for k, v in node.items() if isinstance(v, str) and SHA256_RE.match(v) and "sha256" in k.lower()), None)
                path = next((node[k] for k in ("path", "file", "filename", "relpath") if isinstance(node.get(k), str) and node[k]), None)
                if path is None and isinstance(parent_key, str) and ("/" in parent_key or "." in parent_key):
                    path = parent_key
                if sha and path:
                    add(path if path.startswith(d) else d + path, sha, next((node[k] for k in ("bytes", "size") if isinstance(node.get(k), int)), None), label=path)
                for k, v in node.items():
                    walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    walk(v, parent_key)

        walk(doc)
    return ent


def classify(path: str) -> str:
    if MANIFEST_RE.search(path):
        return "manifest"
    if DOC_RE.search(path):
        return "card_or_doc"
    if SIDECAR_RE.search(path):
        return "sidecar"
    return "other"


def check_code_file(code_root: str | None, entry: dict, current_manifest: dict | None) -> dict:
    """The committed result file a results manifest describes, checked in the checkout; when the checkout's own copy of
    the manifest records a later state of the file, both records are reported."""
    if not code_root:
        return {"status": "no checkout given"}
    rel = "results/revision_october/" + entry["name"]
    p = os.path.join(code_root, *rel.split("/"))
    if not os.path.exists(p):
        return {"status": "absent in the checkout", "path": rel}
    with open(p, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1) if raw else 0
    rows = max(lines - 1, 0)
    out = {"path": rel, "recorded": {"sha256": entry["sha256"], "rows": entry["rows"]}, "checkout": {"sha256": sha, "rows": rows}}
    if sha == entry["sha256"]:
        out["status"] = "verified"
    elif hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == entry["sha256"]:
        out["status"] = "verified after newline normalisation"
    else:
        cur = ((current_manifest or {}).get("committed_files") or {}).get(entry["name"]) or {}
        if cur.get("sha256") == sha:
            out["status"] = "superseded: the checkout's manifest records the later state of this file"
            out["checkout_manifest"] = {"sha256": cur.get("sha256"), "rows": cur.get("rows"),
                                        "updates": sorted(k for k in (current_manifest or {}) if str(k).startswith("update"))}
        else:
            out["status"] = "differs"
    return out


def reconcile(files: list[dict], manifests: dict, digest_cache: dict | None = None, code_root: str | None = None,
              checkout_manifests: dict | None = None) -> dict:
    """manifests: {manifest path: manifest_entries(...)}; digest_cache: {path: sha256} of regular files already fetched
    (an entry whose digest cannot be compared is reported as unverified, not as a match). Per manifest: present,
    digest_verified, listed_without_digest, digest_unverified, missing, not_published, digest_mismatch, size_mismatch,
    absent_ok / absent_present, code_files, views; then the files no manifest covers, by class and top-level directory."""
    by_path = {f["path"]: f for f in files}
    paths_sorted = sorted(by_path)
    digest_cache = digest_cache or {}
    covered: set = set()
    declared_absent = [a for ent in manifests.values() for a in ent["absent"] if not a["path"].endswith("/")]

    def declared_for(path: str):
        base = path.rsplit("/", 1)[-1]
        x = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
        for a in declared_absent:
            if a["path"].rsplit("/", 1)[-1] != base:
                continue
            y = a["path"].rsplit("/", 1)[0] + "/" if "/" in a["path"] else ""
            if y.startswith(x) or x.startswith(y):
                return a
        return None

    def under(prefix: str) -> list[dict]:
        i = bisect.bisect_left(paths_sorted, prefix)
        out = []
        while i < len(paths_sorted) and paths_sorted[i].startswith(prefix):
            out.append(by_path[paths_sorted[i]])
            i += 1
        return out

    report: dict = {}
    all_run_ids = {r for ent in manifests.values() for r in ent["run_ids"]}
    for mpath, ent in manifests.items():
        res: dict = {"schema": ent["schema"], "entries": len(ent["files"]), "present": 0, "present_bytes": 0, "digest_verified": 0,
                     "listed_without_digest": 0, "digest_unverified": [], "missing": [], "not_published": [], "digest_mismatch": [],
                     "size_mismatch": [], "absent_ok": 0, "absent_present": [], "code_files": {}, "views": {}, "notes": list(ent["notes"])}
        for e in ent["files"]:
            f = by_path.get(e["path"])
            if f is None:
                a = declared_for(e["path"])
                if a:
                    res["not_published"].append({"entry": e["label"], "declared_by": a["declared_by"]})
                else:
                    res["missing"].append(e["label"])
                continue
            covered.add(e["path"])
            res["present"] += 1
            res["present_bytes"] += f["size"]
            if e["bytes"] is not None and e["bytes"] != f["size"]:
                res["size_mismatch"].append({"entry": e["label"], "manifest": e["bytes"], "repository": f["size"]})
            if not e["sha256"]:
                res["listed_without_digest"] += 1
                continue
            s = f["sha256"] or digest_cache.get(e["path"])
            if s is None:
                res["digest_unverified"].append(e["label"])
            elif s == e["sha256"]:
                res["digest_verified"] += 1
            else:
                res["digest_mismatch"].append({"entry": e["label"], "path": e["path"], "manifest": e["sha256"], "repository": s})
        for a in ent["absent"]:
            present = [x["path"] for x in under(a["path"])] if a["path"].endswith("/") else ([a["path"]] if a["path"] in by_path else [])
            if present:
                res["absent_present"].append({"entry": a["label"], "present": present[:5]})
            else:
                res["absent_ok"] += 1
        cur = (checkout_manifests or {}).get(mpath.rsplit("/", 1)[-1])
        for c in ent["code_files"]:
            res["code_files"][c["name"]] = check_code_file(code_root, c, cur)
        for v in ent["views"]:
            fl = under(v["dir"])
            types: dict = {}
            for f in fl:
                base = f["path"].rsplit("/", 1)[-1]
                t = base.split(".", 1)[1] if "." in base else ""
                types[t] = types.get(t, 0) + 1
            digest = hashlib.sha256("\n".join(v["shard_lines"]).encode("utf-8")).hexdigest()
            pairs = {"files": (v["files"], len(fl)), "bytes": (v["bytes"], sum(f["size"] for f in fl)), "shards": (v["shards"], len(v["shard_lines"])),
                     "types": (v["types"], types), "shard_digest": (v["shard_digest"], digest)}
            res["views"][v["view"]] = {k: {"manifest": a, "repository": b, "agree": a == b} for k, (a, b) in pairs.items()}
        if ent["not_in_this_part"]:
            gone = [r for r in ent["not_in_this_part"] if r not in all_run_ids]
            res["not_in_this_part"] = {"declared": len(ent["not_in_this_part"]), "found_in_other_parts": len(ent["not_in_this_part"]) - len(gone),
                                       "missing_everywhere": gone[:50]}
        res["views_disagreeing"] = sorted(v for v, c in res["views"].items() if not all(x["agree"] for x in c.values()))
        for k in ("missing", "not_published", "digest_mismatch", "size_mismatch", "digest_unverified", "absent_present"):
            res[k + "_count"] = len(res[k])
            res[k] = res[k][:50]
        report[mpath] = res
    extras = [f for f in files if f["path"] not in covered]
    by_class: dict = {}
    for f in extras:
        c = by_class.setdefault(classify(f["path"]), {"files": 0, "bytes": 0, "top_level": {}})
        c["files"] += 1
        c["bytes"] += f["size"]
        top = f["path"].split("/", 1)[0] if "/" in f["path"] else "."
        c["top_level"][top] = c["top_level"].get(top, 0) + 1
    other = sorted(f["path"] for f in extras if classify(f["path"]) == "other")
    return {"manifests": report, "covered_files": len(covered), "covered_bytes": sum(by_path[p]["size"] for p in covered),
            "uncovered": by_class, "uncovered_other_paths": other[:300], "uncovered_other_count": len(other)}


def problems(rec: dict) -> dict:
    """Per manifest, the counts that make the run fail (missing files, digest or size mismatches, declared-absent files
    that exist, view aggregates that disagree, code files that differ, run ids found in no part)."""
    out = {}
    for m, r in rec["manifests"].items():
        p = {"missing": r["missing_count"], "digest_mismatch": r["digest_mismatch_count"], "size_mismatch": r["size_mismatch_count"],
             "absent_present": r["absent_present_count"], "views_disagreeing": len(r["views_disagreeing"]),
             "code_files_differ": sum(1 for c in r["code_files"].values() if c.get("status") in ("differs", "absent in the checkout")),
             "run_ids_missing_everywhere": len((r.get("not_in_this_part") or {}).get("missing_everywhere") or [])}
        p = {k: v for k, v in p.items() if v}
        if p:
            out[m] = p
    return out


def run(kind: str, repo: str, revision: str, out: str, hub: Hub, code_root: str | None, fetch_regular: bool = True) -> dict:
    if hub.offline:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit(f"--offline needs the full commit id as the {kind} revision")
        cached = hub.cached(kind, revision, "info.json")
        info = json.loads(cached) if cached else {"commit": revision, "last_modified": None, "private": None, "gated": None}
    else:
        info = resolve_revision(kind, repo, revision, hub.get)
        hub.store(kind, info["commit"], json.dumps(info).encode("utf-8"), "info.json")
    commit = info["commit"]
    cached = hub.cached(kind, commit, "tree.json")
    if cached:
        tree = json.loads(cached)
        files, pages, listed_utc = tree["files"], tree["pages"], tree["listed_utc"]
    else:
        files, pages = list_tree(kind, repo, commit, hub.get)
        listed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        hub.store(kind, commit, json.dumps({"files": files, "pages": pages, "listed_utc": listed_utc}).encode("utf-8"), "tree.json")
    inv = write_inventory(os.path.join(out, f"inventory_{kind}_{commit[:7]}.csv.gz"), files)
    base = f"{HUB}/{'datasets/' if kind == 'dataset' else ''}{repo}/resolve/{commit}/"

    def get_file(path: str) -> bytes:
        return hub.get(base + urllib.parse.quote(path))[0]

    manifests, manifest_digests = {}, {}
    for f in files:
        if not MANIFEST_RE.search(f["path"]) or f["size"] > SMALL * 4:
            continue
        raw = hub.cached(kind, commit, "manifests", *f["path"].split("/"))
        if raw is None:
            raw = get_file(f["path"])
            hub.store(kind, commit, raw, "manifests", *f["path"].split("/"))
        manifest_digests[f["path"]] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        try:
            ent = manifest_entries(json.loads(raw.decode("utf-8")), f["path"])
        except ValueError:
            ent = {"schema": "not JSON", "files": [], "absent": [], "code_files": [], "views": [], "run_ids": [], "not_in_this_part": [], "notes": ["not JSON"]}
        manifests[f["path"]] = ent
        manifest_digests[f["path"]]["schema"] = ent["schema"]
    cached = hub.cached(kind, commit, "digests.json")
    digests: dict = json.loads(cached) if cached else {}
    by_path = {f["path"]: f for f in files}
    todo = sorted({e["path"] for ent in manifests.values() for e in ent["files"]
                   if e["sha256"] and e["path"] in by_path and not by_path[e["path"]]["lfs"] and by_path[e["path"]]["size"] <= SMALL and e["path"] not in digests})
    fetch_failures: list = []
    if todo and fetch_regular and not hub.offline:
        print(f"{kind}: fetching {len(todo)} regular files to verify their digests", file=sys.stderr)

        def one(p: str):
            try:
                return p, hashlib.sha256(get_file(p)).hexdigest()
            except Exception as ex:  # noqa: BLE001 - reported per file
                return p, "ERROR " + type(ex).__name__ + ": " + str(ex)[:120]

        done = 0
        with ThreadPoolExecutor(max_workers=hub.threads) as ex:
            for p, s in ex.map(one, todo):
                done += 1
                if s.startswith("ERROR"):
                    fetch_failures.append({"path": p, "error": s[6:]})
                else:
                    digests[p] = s
                if done % 500 == 0:
                    hub.store(kind, commit, json.dumps(digests).encode("utf-8"), "digests.json")
                    print(f"{kind}: {done}/{len(todo)} regular files hashed", file=sys.stderr)
        hub.store(kind, commit, json.dumps(digests).encode("utf-8"), "digests.json")
    checkout_manifests = {}
    if code_root:
        for name in ("manifest.part-A.json", "manifest.part-B.json"):
            p = os.path.join(code_root, "results", "revision_october", name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    checkout_manifests[name] = json.load(fh)
    rec = reconcile(files, manifests, digests, code_root, checkout_manifests)
    rec["regular_files_hashed"] = len(digests)
    rec["regular_files_not_fetched"] = len(todo) if (hub.offline or not fetch_regular) else len(fetch_failures)
    rec["fetch_failures"] = fetch_failures[:50]
    top: dict = {}
    for f in files:
        t = f["path"].split("/", 1)[0] if "/" in f["path"] else "."
        top.setdefault(t, {"files": 0, "bytes": 0})
        top[t]["files"] += 1
        top[t]["bytes"] += f["size"]
    summary = {"kind": kind, "repo": repo, "requested_revision": revision, **info, "files": len(files), "bytes": sum(f["size"] for f in files),
               "lfs_files": sum(1 for f in files if f["lfs"]), "pages": pages, "listed_utc": listed_utc,
               "reconciled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "top_level": top, "inventory": inv, "manifests": manifest_digests,
               "hub": {"requests": hub.requests, "retries_429": hub.retries_429, "other_retries": hub.other_retries, "offline": hub.offline}}
    return {"summary": summary, "reconciliation": rec}


def git_head(code_root: str | None) -> str | None:
    if not code_root:
        return None
    try:
        r = subprocess.run(["git", "-C", code_root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="ru1ch3n/PDE-OBS")
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--kinds", nargs="*", default=("dataset", "model"), choices=("dataset", "model"))
    ap.add_argument("--cache-dir", default=None, help="keep the listing, the manifests and the digests of fetched regular files here")
    ap.add_argument("--offline", action="store_true", help="use the cache only (revisions must be full commit ids)")
    ap.add_argument("--no-fetch", action="store_true", help="do not fetch regular files; their digests stay unverified")
    ap.add_argument("--pace", type=float, default=PACE, help="seconds between request starts")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--code-root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
                    help="checkout of the benchmark code (its results/revision_october files are checked against the results manifests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hub = Hub(a.cache_dir, a.offline, a.pace, a.threads)
    summaries, recs = {}, {}
    for kind in a.kinds:
        r = run(kind, a.repo, a.dataset_revision if kind == "dataset" else a.model_revision, a.out, hub, a.code_root, not a.no_fetch)
        summaries[kind], recs[kind] = r["summary"], r["reconciliation"]
        s = r["summary"]
        print(f"{kind}: commit {s['commit'][:7]}, {s['files']} files, {s['bytes']} bytes, {s['pages']} pages, {len(s['manifests'])} manifests; "
              f"covered {r['reconciliation']['covered_files']} files; problems {json.dumps(problems(r['reconciliation']))}", file=sys.stderr)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=1)
    with open(os.path.join(a.out, "reconciliation.json"), "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=1)
    bindings = {"schema": "pdeobs-release-bindings/v2", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repositories": {},
                "code_repository": {"git_head": git_head(a.code_root), "results_files": {}}, "problems": {}}
    for kind, s in summaries.items():
        bindings["repositories"][kind] = {"repo": s["repo"], "commit": s["commit"], "requested_revision": s["requested_revision"],
                                          "last_modified": s["last_modified"], "listed_utc": s["listed_utc"], "files": s["files"], "bytes": s["bytes"],
                                          "lfs_files": s["lfs_files"], "inventory": s["inventory"], "manifests": s["manifests"]}
        for m, r in recs[kind]["manifests"].items():
            for name, c in r["code_files"].items():
                bindings["code_repository"]["results_files"].setdefault(name, {})[m] = c
        bindings["problems"][kind] = problems(recs[kind])
    with open(os.path.join(a.out, "release_bindings.json"), "w", encoding="utf-8") as fh:
        json.dump(bindings, fh, indent=1)
    bad = {k: v for k, v in bindings["problems"].items() if v}
    print(json.dumps({"out": a.out, "problems": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Unauthenticated access, integrity and execution checks of the public deposits.

Three separately reported checks:
    access     can an unauthenticated client reach the repository API and the declared files?  Plain HTTPS
               requests with no token, no cookies and no client library; every status code is recorded.
    integrity  do downloaded files match the frozen digests (release manifests)?  Runs only when access succeeds.
    execution  can the released code load one stationary and one temporal model and score the full 200-identity
               blocks?  Runs only when the artifacts of those two evaluations were downloaded and verified
               (``tools/verify_archived_predictions.py --device cpu``).  A smoke test of two evaluations, labelled as such.

An anonymity inspection of the local release cards and of any downloaded README is always performed.
Output: ``access_check.json`` in --out.  Nothing here uses credentials; if a credential is found in the
environment it is ignored and the report says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_yaml, utc_now, write_json  # noqa: E402

CHECK_VERSION = "pdeobs-revision-access-check/20260926-v1"
# Generic patterns only (private machine paths).  Author-specific strings are supplied at run time through
# --patterns-file (one regular expression per line) so that they never enter this public repository.
IDENTIFYING = re.compile(r"(/lustre/|/gpfs/|/scratch/|/home/[a-z]|/Users/[a-z])", re.I)
USER_AGENT = "pdeobs-anonymous-access-check/1"


def request(url: str, *, head: bool = False, limit: int = 1 << 20) -> dict[str, Any]:
    req = urllib.request.Request(url, method="HEAD" if head else "GET", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = b"" if head else response.read(limit)
            return {"url": url, "status": response.status, "final_url": response.geturl(), "content_length": response.headers.get("Content-Length"),
                    "etag": response.headers.get("ETag") or response.headers.get("X-Linked-Etag"), "body": body}
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "final_url": exc.geturl(), "error": exc.reason, "body": b""}
    except Exception as exc:  # noqa: BLE001 - network errors are results here
        return {"url": url, "status": None, "error": f"{type(exc).__name__}: {exc}", "body": b""}


def download(url: str, dest: Path, expected_sha256: str | None) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as response, dest.open("wb") as handle:
            while chunk := response.read(8 << 20):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "downloaded": False}
    result = {"url": url, "status": 200, "downloaded": True, "bytes": size, "sha256": digest.hexdigest()}
    if expected_sha256:
        result["matches_manifest"] = digest.hexdigest() == expected_sha256
    return result


def inspect_text(label: str, text: str) -> dict[str, Any]:
    hits = sorted({m.group(0).lower() for m in IDENTIFYING.finditer(text)})
    return {"source": label, "characters": len(text), "identifying_strings": hits, "clean": not hits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposits", type=Path, default=REPO / "configs/revision/public_deposits.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "results/revision_v2")
    parser.add_argument("--download-dir", type=Path, default=None, help="where smoke-test artifacts are stored (fresh directory)")
    parser.add_argument("--smoke", action="store_true", help="download and verify the two smoke-test evaluations when access succeeds")
    parser.add_argument("--patterns-file", type=Path, default=None, help="extra identifying patterns (kept outside the repository)")
    args = parser.parse_args()
    global IDENTIFYING
    if args.patterns_file:
        extra = [line.strip() for line in args.patterns_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        IDENTIFYING = re.compile("(" + "|".join([IDENTIFYING.pattern] + extra) + ")", re.I)
    deposits = load_yaml(args.deposits)
    ignored = [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN") if os.environ.get(k)]
    for key in ignored:
        os.environ.pop(key, None)
    report: dict[str, Any] = {"schema": CHECK_VERSION, "generated_utc": utc_now(), "credentials": {"environment_tokens_ignored": ignored, "token_file_used": False,
                                                                                                    "client_library": "none (urllib), no cookies"},
                              "access": {}, "integrity": {}, "execution": {"status": "not_run", "reason": "requires access and integrity to succeed"}, "anonymity": []}
    for repo in deposits["repositories"]:
        kind, rid = repo["type"], repo["id"]
        api = f"https://huggingface.co/api/{'datasets' if kind == 'dataset' else 'models'}/{rid}"
        base = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{rid}/resolve/{repo.get('revision', 'main')}"
        entry: dict[str, Any] = {"type": kind, "declared_role": repo.get("role"), "api": request(api)}
        entry["api"].pop("body", None)
        entry["files"] = {}
        for name in repo.get("files", []):
            r = request(f"{base}/{name}", head=True)
            r.pop("body", None)
            entry["files"][name] = r
        readme = request(f"{base}/README.md")
        if readme["status"] == 200:
            report["anonymity"].append(inspect_text(f"{rid}/README.md (downloaded)", readme["body"].decode("utf-8", "replace")))
        entry["readme_status"] = readme["status"]
        entry["reachable_unauthenticated"] = entry["api"]["status"] == 200 and all(f["status"] in (200, 302) for f in entry["files"].values())
        report["access"][rid] = entry
    for card in deposits.get("local_cards", []):
        path = REPO / card if not Path(card).is_absolute() else Path(card)
        if path.exists():
            report["anonymity"].append(inspect_text(str(card), path.read_text(encoding="utf-8", errors="replace")))
    reachable = all(e["reachable_unauthenticated"] for e in report["access"].values())
    report["summary"] = {"all_repositories_reachable_unauthenticated": reachable,
                         "statuses": {rid: e["api"]["status"] for rid, e in report["access"].items()},
                         "anonymity_clean": all(a["clean"] for a in report["anonymity"])}
    if reachable and args.smoke:
        root = args.download_dir or (args.out / "access_smoke")
        root.mkdir(parents=True, exist_ok=True)
        results = {}
        for item in deposits.get("smoke_test", []):
            repo = next(r for r in deposits["repositories"] if r["id"] == item["repository"])
            base = f"https://huggingface.co/{'datasets/' if repo['type'] == 'dataset' else ''}{repo['id']}/resolve/{repo.get('revision', 'main')}"
            results[item["path"]] = download(f"{base}/{item['path']}", root / item["path"], item.get("sha256"))
        report["integrity"] = {"downloads": results, "all_match": all(r.get("matches_manifest", False) for r in results.values())}
        report["execution"] = {"status": "not_run", "reason": "run tools/verify_archived_predictions.py --device cpu on the downloaded roots; see docs/prediction_verification.md"}
    elif not reachable:
        report["integrity"] = {"status": "not_run", "reason": "unauthenticated access failed"}
    write_json(args.out / "access_check.json", report)
    print(json.dumps(report["summary"], indent=1))
    return 0 if reachable else 2


if __name__ == "__main__":
    raise SystemExit(main())

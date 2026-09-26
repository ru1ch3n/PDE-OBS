"""Run the explicitly scoped portable tests without changing the original tests.

Use new output and temporary directories outside the source tree. Local execution
receipts include caller-supplied filesystem paths; review them before publication.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
from datetime import datetime, timezone


class Tee:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


class TestReports:
    def __init__(self):
        self.collected = []
        self.collection_errors = []
        self.collection_skips = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({"nodeid": report.nodeid, "detail": str(report.longrepr)})
        if report.skipped:
            self.collection_skips.append({"nodeid": report.nodeid, "detail": str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        item = {"nodeid": report.nodeid, "phase": report.when,
                "outcome": report.outcome, "duration_seconds": report.duration}
        if report.failed or report.skipped:
            item["detail"] = str(report.longrepr)
        self.reports.append(item)

    def counts(self):
        counts = {"collected": len(self.collected), "passed": 0, "failed": 0,
                  "skipped": 0, "not_run": 0,
                  "collection_errors": len(self.collection_errors),
                  "collection_skips": len(self.collection_skips)}
        for nodeid in self.collected:
            phases = [item for item in self.reports if item["nodeid"] == nodeid]
            if any(item["outcome"] == "failed" for item in phases):
                counts["failed"] += 1
            elif any(item["outcome"] == "skipped" for item in phases):
                counts["skipped"] += 1
            elif any(item["phase"] == "call" and item["outcome"] == "passed" for item in phases):
                counts["passed"] += 1
            else:
                counts["not_run"] += 1
        return counts


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(path, parent):
    return path == parent or parent in path.parents


def peak_memory():
    """Return this process's peak resident memory; child processes are excluded."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                             ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"peak_rss_bytes": counters.PeakWorkingSetSize, "scope": "runner_process_only"}
        return {"peak_rss_bytes": None, "scope": "runner_process_only", "error": ctypes.get_last_error()}
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(value if sys.platform == "darwin" else value * 1024),
            "scope": "runner_process_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; must not overlap the source or basetemp.")
    parser.add_argument("--basetemp", type=Path, required=True,
                        help="New pytest temporary directory; existing paths are refused to prevent deletion.")
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    basetemp = args.basetemp.resolve()
    for required in ("src/pdeobs/__init__.py", "pyproject.toml", "tests"):
        if not (source / required).exists():
            parser.error(f"Not a source release: missing {required}")
    for name, path in (("output-dir", output), ("basetemp", basetemp)):
        if path.exists() or path.is_symlink():
            parser.error(f"{name} already exists; choose a fresh path, nothing was deleted: {path}")
        if path == Path(path.anchor) or within(path, source) or within(source, path):
            parser.error(f"{name} must be outside and not an ancestor of the source tree")
    if within(output, basetemp) or within(basetemp, output):
        parser.error("output-dir and basetemp must be separate, non-nested directories")

    scope_path = Path(__file__).resolve().with_name("test_scope.json")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("schema_version") != "pdeobs-portable-core-test-scope-v1":
        parser.error("Unsupported test selection schema")
    files = scope["selected_files"]
    nodes = scope["selected_nodes"]
    selectors = files + nodes
    if len(selectors) != len(set(selectors)):
        parser.error("The selection contains duplicate selectors")
    if scope["selection_counts"] != {"whole_files": len(files), "individual_nodes": len(nodes)}:
        parser.error("The manifest selection count does not match its lists")
    for selector in selectors:
        filename = selector.split("::", 1)[0]
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or not filename.startswith("tests/"):
            parser.error(f"Invalid relative test selector: {selector}")
        resolved = (source / filename).resolve(strict=True)
        if not within(resolved, source / "tests") or not resolved.is_file():
            parser.error(f"Selected file is not in the source tests directory: {selector}")
    if any(node.split("::", 1)[0] in files for node in nodes):
        parser.error("An individually selected node is already covered by a whole selected file")

    output.mkdir(parents=True, exist_ok=False)
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    runtime_temp = output / "runtime-temp"
    runtime_temp.mkdir()
    thread_environment = {
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": "",
        "PYTHONPATH": str(source / "src"), "TEMP": str(runtime_temp), "TMP": str(runtime_temp),
        "TMPDIR": str(runtime_temp),
    }
    os.environ.update(thread_environment)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))
    os.chdir(source)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    cpu_started = time.process_time()
    reports = TestReports()
    pytest_args = ["-q", "-ra", "-p", "no:cacheprovider", "--rootdir", str(source),
                   "--confcutdir", str(source / "tests"), "--basetemp", str(basetemp),
                   "--junitxml", str(output / "junit.xml"), *selectors]
    receipt = {
        "schema_version": "pdeobs-portable-core-test-execution-v1",
        "started_utc": started_utc, "source_root": str(source), "output_dir": str(output),
        "basetemp": str(basetemp), "runner_sha256": sha256(Path(__file__).resolve()),
        "scope_sha256": sha256(scope_path), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "selection_counts": scope["selection_counts"], "selectors": selectors,
        "selected_test_file_sha256": {
            path: sha256(source / path) for path in sorted({item.split("::", 1)[0] for item in selectors})
        },
        "pytest_arguments": pytest_args, "thread_environment": thread_environment,
        "excluded_not_executed": {"files": scope["excluded_files"], "nodes": scope["excluded_nodes"]},
        "limitations": ["A source-tree test, not a wheel-only installation test.",
                        "Memory measurement excludes subprocesses.",
                        "Deployment and missing historical-fixture tests are not counted as passed.",
                        "No scientific test assertion or threshold is changed by this runner."],
    }
    exit_code = 3
    with (output / "pytest.log").open("x", encoding="utf-8", newline="\n") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import pytest
                import pdeobs
                package_path = Path(pdeobs.__file__).resolve()
                if not within(package_path, source / "src" / "pdeobs"):
                    raise RuntimeError("pdeobs imported from outside the selected source tree")
                receipt["package_import_path"] = str(package_path)
                receipt["versions"] = {}
                for name in ("numpy", "scipy", "h5py", "PyYAML", "pytest", "torch"):
                    try:
                        receipt["versions"][name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        receipt["versions"][name] = None
                if importlib.util.find_spec("torch") is not None:
                    import torch
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                    receipt["torch_threads"] = {"intraop": torch.get_num_threads(),
                                                "interop": torch.get_num_interop_threads()}
                print(f"Running {len(files)} complete files and {len(nodes)} specific test nodes on CPU.")
                exit_code = int(pytest.main(pytest_args, plugins=[reports]))
            except Exception:
                receipt["runner_error"] = traceback.format_exc()
                traceback.print_exc()
            finally:
                receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
                receipt["duration_seconds"] = time.perf_counter() - started
                receipt["cpu_seconds"] = time.process_time() - cpu_started
                receipt["memory"] = peak_memory()
                receipt["exit_code"] = exit_code
                receipt["counts"] = reports.counts()
                receipt["collected_nodes"] = reports.collected
                receipt["collection_errors"] = reports.collection_errors
                receipt["collection_skips"] = reports.collection_skips
                receipt["test_reports"] = reports.reports
                counts = receipt["counts"]
                if exit_code != 0 or counts["failed"] or counts["collection_errors"] or counts["not_run"]:
                    receipt["status"] = "failed_or_incomplete"
                elif counts["skipped"] or counts["collection_skips"]:
                    receipt["status"] = "passed_selected_tests_with_skips"
                else:
                    receipt["status"] = "passed_selected_tests"
                print(json.dumps({"status": receipt["status"], "counts": counts, "exit_code": exit_code}))
    receipt["log_sha256"] = sha256(output / "pytest.log")
    if (output / "junit.xml").is_file():
        receipt["junit_sha256"] = sha256(output / "junit.xml")
    with (output / "execution.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Execution receipt: {output / 'execution.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Actual no-PyTorch environment acceptance, not a mocked import test.

Run with a base-only installation's interpreter. This intentionally refuses to
run if torch is installed. New explicit data/output roots are required. Example:

    python -B tests/check_base_only.py --data-root DATA --output-root RUNS

The optional --expected-source-root flag is only for a source-patch regression;
omit it when validating an independently installed wheel outside its source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-source-root", type=Path)
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output_root.resolve()
    if data.exists() or output.exists():
        parser.error("use new data/output roots; prior evidence is never overwritten")
    if data == output or data in output.parents or output in data.parents:
        parser.error("data and output roots must be separate and non-nested")
    output.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    os.environ.update(env)
    started = time.perf_counter()
    receipt = {"schema_version": "pdeobs-real-base-only-acceptance-v1", "status": "running",
               "python": sys.executable, "device": "cpu", "threads": 1,
               "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "assertions": [], "workflows": [], "scope": "real no-Torch interpreter, not mocked imports"}
    exit_code = 1
    try:
        if importlib.util.find_spec("torch") is not None:
            raise RuntimeError("This test requires a genuinely torch-free environment")
        try:
            importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError("A torch distribution is installed; no-Torch test refused")
        receipt["assertions"].append("torch module and distribution absent")
        from pdeobs import __version__
        import pdeobs
        import pdeobs.methods
        import pdeobs.runner
        import pdeobs.evaluation
        import pdeobs.physical_inputs
        from pdeobs.methods import create_method
        from pdeobs.methods.paper_official import OfficialPaperUNet, OfficialPaperFNO2d, OfficialPaperCNO2d
        package = Path(pdeobs.__file__).resolve()
        if args.expected_source_root:
            required = args.expected_source_root.resolve() / "src" / "pdeobs"
            if required not in package.parents:
                raise RuntimeError(f"Imported package outside expected candidate: {package}")
        receipt.update(package_version=__version__, import_path=str(package))
        receipt["assertions"].append("core, methods, runner, evaluation and physical adapters imported")
        for name in ("zero", "mean", "nearest", "bilinear", "rbf", "persistence", "rbf_persistence", "gappy_pod", "gappy_pod_dmd"):
            create_method(name)
        receipt["assertions"].append("nine existing classical constructors available without torch")
        common = dict(upstream_root="unavailable-checkout", upstream_revision="not-loaded",
                      source_file_sha256="not-loaded", paper_pde="poisson")
        official = [(OfficialPaperUNet, dict(channels=32)),
                    (OfficialPaperFNO2d, dict(width=16, modes=16, layers=5, padding=0)),
                    (OfficialPaperCNO2d, dict(layers=3, channels=16, neck_residuals=6, level_residuals=4))]
        for cls, settings in official:
            try:
                cls(**common, **settings)
            except ImportError as exc:
                if "PyTorch" not in str(exc):
                    raise
            else:
                raise RuntimeError(f"{cls.__name__} silently constructed without torch")
        try:
            create_method("unet", width=4)
        except ImportError as exc:
            if "pdeobs[train]" not in str(exc):
                raise
        else:
            raise RuntimeError("neural method silently constructed without torch")
        receipt["assertions"].append("all official adapters and compact UNet fail explicitly at construction; no fallback")
        for example in ("E1", "E2", "E3", "E4", "E6"):
            command = [sys.executable, "-B", "-m", "pdeobs", "demo", "--example", example,
                       "--data-root", str(data), "--output-root", str(output / "workflows"), "--threads", "1"]
            before = time.perf_counter()
            child = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", check=False)
            (output / f"{example}.stdout.log").write_text(child.stdout, encoding="utf-8")
            (output / f"{example}.stderr.log").write_text(child.stderr, encoding="utf-8")
            item = {"example": example, "command": command, "exit_code": child.returncode,
                    "duration_seconds": time.perf_counter() - before}
            receipt["workflows"].append(item)
            if child.returncode:
                raise RuntimeError(f"{example} failed in genuine base-only interpreter")
            workflow = json.loads((output / "workflows" / example / "receipt.json").read_text(encoding="utf-8"))
            if workflow["status"] != "passed":
                raise RuntimeError(f"{example} returned non-passing workflow receipt")
            item["status"] = "passed"
            item["peak_process_rss_bytes"] = workflow["peak_process_rss_bytes"]
        receipt.update(status="passed", exit_code=0)
        exit_code = 0
    except Exception:
        receipt.update(status="failed", exit_code=1, error=traceback.format_exc())
        traceback.print_exc()
    finally:
        receipt["duration_seconds"] = time.perf_counter() - started
        with (output / "base-only-receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        print(json.dumps(receipt, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

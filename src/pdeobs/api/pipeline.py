"""Versioned pipeline configuration and one-call execution (``api.run`` / ``pdeobs easy run``).

The same resolver serves Python objects, YAML and JSON so that every entry point
produces the identical resolved scientific configuration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from . import specs as S

STAGES = ("data", "observation", "model", "train", "predict", "evaluate")
_TOP_KEYS = {"schema_version", "name", "seed", "device", "out", "task", "data", "observation", "model", "train", "predict",
             "evaluate", "stages"}


def load_pipeline_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        config = json.loads(json.dumps(dict(source), default=str))
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            config = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            config = json.loads(text)
        else:
            raise ValueError("pipeline configs are YAML (.yaml/.yml) or JSON (.json)")
    if not isinstance(config, dict):
        raise ValueError("pipeline config must be a mapping")
    version = config.get("schema_version", S.PIPELINE_CONFIG_VERSION)
    if version != S.PIPELINE_CONFIG_VERSION:
        raise ValueError(f"pipeline schema {version!r} is not {S.PIPELINE_CONFIG_VERSION!r}")
    unknown = set(config) - _TOP_KEYS
    if unknown:
        raise ValueError(f"pipeline config has unknown top-level keys {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}")
    config.setdefault("schema_version", S.PIPELINE_CONFIG_VERSION)
    return config


def validate_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every stage without generating data, training or writing (cheap checks only)."""
    from .models import check_grid_compatibility, resolve_model
    from .observation import make_observation
    from .train import Budget

    cfg = load_pipeline_config(config)
    resolved: dict[str, Any] = {"schema_version": S.PIPELINE_CONFIG_VERSION, "api_version": S.API_VERSION, "name": cfg.get("name"),
                                "seed": int(cfg.get("seed", 0)), "device": cfg.get("device", "auto"), "out": cfg.get("out"),
                                "stages": list(cfg.get("stages") or [s for s in STAGES if s in cfg])}
    requested_stages = cfg.get("stages", [s for s in STAGES if s in cfg])
    if not isinstance(requested_stages, list) or not requested_stages:
        raise ValueError("stages must be a non-empty list of known pipeline stages")
    if any(not isinstance(s, str) or s not in STAGES for s in requested_stages):
        raise ValueError(f"unknown pipeline stage; valid stages: {list(STAGES)}")
    if len(set(requested_stages)) != len(requested_stages):
        raise ValueError("pipeline stages must not contain duplicates")
    if any(stage not in cfg for stage in requested_stages):
        raise ValueError("each requested stage needs its corresponding configuration section")
    # Execution is in canonical dependency order; preserve that order in the plan.
    resolved["stages"] = [stage for stage in STAGES if stage in requested_stages]
    if "train" in requested_stages and not {"data", "model", "observation"}.issubset(requested_stages):
        raise ValueError("train execution requires data, observation and model stages")
    task = cfg.get("task")
    data = cfg.get("data")
    if data is not None:
        data = dict(data)
        source = data.get("source", "local" if "path" in data else "generate")
        if source == "generate":
            required = {"pde", "out"}
            missing = required - set(data)
            if missing:
                raise ValueError(f"data.generate needs {sorted(missing)}")
            pde = data["pde"]
            static = pde in S.STATIC_PDES
            if not static and not data.get("time_steps"):
                raise ValueError(f"data.time_steps is required for temporal PDE {pde}")
            res = data.get("resolution", 32)
            resolution = (res, res) if isinstance(res, int) else tuple(res)
            resolved["data"] = {"source": "generate", **data, "resolution": list(resolution), "estimated_samples": int(data.get("num_samples", 4)),
                                "estimated_bytes": int(data.get("num_samples", 4)) * int(resolution[0]) * int(resolution[1]) * 4 * (int(data.get("time_steps") or 1) + 2)}
        elif source in {"local", "manifest"}:
            if "path" not in data:
                raise ValueError("data.local needs 'path'")
            resolved["data"] = {"source": source, **data}
            if Path(data["path"]).exists():
                from .data import load_dataset
                handle = load_dataset(data["path"], source=source, manifest=data.get("manifest"))
                resolved["data"]["inspected"] = handle.to_dict()
                resolution = tuple(handle.resolution)
            else:
                resolved["data"]["inspected"] = None
                resolution = None
        elif source == "download":
            raise ValueError("no verified public download endpoint is registered; use generate or local")
        else:
            raise ValueError(f"unknown data.source {source!r}")
    else:
        resolution = None
    if "observation" in cfg:
        obs = make_observation(cfg["observation"]) if not isinstance(cfg["observation"], str) else make_observation(cfg["observation"])
        resolved["observation"] = obs.to_dict()
    if "model" in cfg:
        if task is None:
            raise ValueError("model resolution needs the pipeline 'task'")
        m = dict(cfg["model"]) if isinstance(cfg["model"], Mapping) else {"name": cfg["model"]}
        mc = resolve_model(m, task=task, preset=m.get("preset"), pde=(data or {}).get("pde") if data else None)
        resolved["model"] = mc.to_dict()
        if resolution is not None:
            problems = check_grid_compatibility(mc, (int(resolution[0]), int(resolution[1])))
            if problems:
                raise ValueError("; ".join(problems))
    if "train" in cfg:
        tr = dict(cfg["train"])
        budget = Budget.parse(tr.get("budget"))
        if "model" not in cfg or "observation" not in cfg or data is None:
            raise ValueError("train stage needs data, observation and model sections")
        resolved["train"] = {**tr, "budget": budget.to_dict()}
    if "predict" in cfg:
        pr = dict(cfg["predict"])
        if "inputs" not in pr and data is None:
            raise ValueError("predict needs 'inputs' (observation package) or a data section for benchmark mode")
        resolved["predict"] = pr
    if "evaluate" in cfg:
        resolved["evaluate"] = dict(cfg["evaluate"])
    resolved["task"] = task
    return resolved


def plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Human-readable plan: stages, data scale, model, budget; no execution."""
    resolved = validate_pipeline(config)
    lines = [f"pipeline {resolved.get('name') or '<unnamed>'} (schema {resolved['schema_version']})", f"stages: {', '.join(resolved['stages'])}"]
    if "data" in resolved:
        d = resolved["data"]
        if d["source"] == "generate":
            lines.append(f"data: generate {d['pde']} {d.get('boundary', 'periodic')}/{d.get('setting', 'smooth_grf')}/{d.get('regime', 'low')} "
                         f"x{d['estimated_samples']} at {d['resolution']} T={d.get('time_steps') or 1} (~{d['estimated_bytes'] / 1e6:.1f} MB) -> {d['out']}")
        else:
            lines.append(f"data: {d['source']} {d['path']} " + (f"({d['inspected']['sample_count']} records, {d['inspected']['resolution']}, T={d['inspected']['time_steps']})" if d.get("inspected") else "(not present yet)"))
    if "observation" in resolved:
        o = resolved["observation"]
        lines.append(f"observation: {o['namespace']}:{o['name']} -> {o['mask_config']}")
    if "model" in resolved:
        m = resolved["model"]
        lines.append(f"model: {m['name']} ({m['origin']}) {m['kwargs']} task={m['task']}")
    if "train" in resolved:
        lines.append(f"train: budget={resolved['train']['budget']} split={resolved['train'].get('split', 'all')} batch_size={resolved['train'].get('batch_size', 'auto')}")
    if "predict" in resolved:
        lines.append(f"predict: {resolved['predict']}")
    if "evaluate" in resolved:
        lines.append(f"evaluate: {resolved['evaluate']}")
    return {"resolved": resolved, "text": "\n".join(lines)}


def run(config: str | Path | Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Execute the declared stages: data -> observation -> model -> train -> predict -> evaluate."""
    from .data import DatasetHandle, InferenceInput, create_dataset, load_dataset
    from .artifacts import load_predictor
    from .evaluate import evaluate, evaluate_records
    from .infer import predict
    from .observation import make_observation
    from .train import train

    resolved = validate_pipeline(config)
    if dry_run:
        return {"status": "validated", "resolved": resolved}
    cfg = load_pipeline_config(config)
    out_root = Path(cfg.get("out") or "pdeobs-run")
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt: dict[str, Any] = {"schema_version": S.PIPELINE_CONFIG_VERSION, "resolved": resolved, "stages": {}, "status": "running"}
    stages = resolved["stages"]
    handle: DatasetHandle | None = None
    obs = None
    artifact = None
    bundle = None
    try:
        if "data" in stages:
            d = dict(cfg["data"])
            source = d.pop("source", "local" if "path" in d else "generate")
            if source == "generate":
                handle = create_dataset(source="generate", **d)
            else:
                handle = load_dataset(d["path"], source=source, manifest=d.get("manifest"), verify=bool(d.get("verify", False)))
            receipt["stages"]["data"] = handle.to_dict()
        if "observation" in stages:
            obs = make_observation(cfg["observation"])
            receipt["stages"]["observation"] = obs.to_dict()
        if "train" in stages:
            tr = dict(cfg["train"])
            m = dict(cfg["model"]) if isinstance(cfg["model"], Mapping) else {"name": cfg["model"]}
            result = train(handle, task=cfg["task"], model={"name": m["name"], "params": m.get("params", {})}, preset=m.get("preset"),
                           observation=obs, out=out_root / tr.get("out", "train"), budget=tr.get("budget"), seed=int(cfg.get("seed", 0)),
                           device=cfg.get("device", "auto"), batch_size=tr.get("batch_size"), split=tr.get("split", "all"),
                           history_steps=int(tr.get("history_steps", 1)), horizon=tr.get("horizon"), training=tr.get("training"),
                           num_workers=int(tr.get("num_workers", 0)), label=tr.get("label"), max_samples=tr.get("max_samples"))
            artifact = result.artifact
            receipt["stages"]["train"] = result.to_dict()
        if "predict" in stages:
            pr = dict(cfg["predict"])
            predictor = load_predictor(pr.get("model") or (artifact.path if artifact else None), device=cfg.get("device", "auto"))
            if "inputs" in pr:
                bundle = predict(predictor, InferenceInput.load(pr["inputs"]), observation="stored", horizon=pr.get("horizon"),
                                 batch_size=int(pr.get("batch_size", 4)), out=out_root / pr.get("out", "predictions.npz"),
                                 free_geometry=bool(pr.get("free_geometry", False)))
            else:
                bundle = predict(predictor, handle, observation=obs, horizon=pr.get("horizon"), batch_size=int(pr.get("batch_size", 4)),
                                 out=out_root / pr.get("out", "predictions.npz"), sample_ids=pr.get("sample_ids"), max_samples=pr.get("max_samples"))
            receipt["stages"]["predict"] = bundle.to_dict()
        if "evaluate" in stages:
            ev = dict(cfg["evaluate"])
            if ev.get("mode", "bundle") == "records":
                predictor = load_predictor(ev.get("model") or (artifact.path if artifact else None), device=cfg.get("device", "auto"))
                report = evaluate_records(predictor, handle, obs, out=out_root / ev.get("out", "evaluation"), sample_ids=ev.get("sample_ids"),
                                          batch_size=int(ev.get("batch_size", 4)), max_samples=ev.get("max_samples"))
            else:
                if bundle is None:
                    raise ValueError("evaluate in bundle mode needs a predict stage in the same run (or use mode: records)")
                report = evaluate(bundle, out=out_root / ev.get("out", "evaluation"))
            receipt["stages"]["evaluate"] = {"status": report.get("status"), "summary": report.get("summary"), "errors": report.get("errors", [])}
            if report.get("status") != "valid":
                raise ValueError(f"strict evaluation was not valid: {report.get('errors', [])}")
        receipt["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - the receipt records the failure and re-raises
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["seconds"] = round(time.perf_counter() - started, 3)
        (out_root / "pipeline_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return receipt

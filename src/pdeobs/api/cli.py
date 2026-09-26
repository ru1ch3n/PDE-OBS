"""``pdeobs easy ...`` command namespace.  Shares the resolver with the Python API."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _parse_value(text: str) -> Any:
    """Parse ``key=value`` values with YAML scalars only (no code execution)."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, (dict, list)):
        raise argparse.ArgumentTypeError(f"parameter values must be scalars, got {text!r}")
    return value


def _kv(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"empty key in {item!r}")
        if key in out:
            raise SystemExit(f"parameter {key!r} given twice")
        out[key] = _parse_value(value.strip())
    return out


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(yaml.safe_dump(json.loads(json.dumps(payload, default=str)), sort_keys=False, allow_unicode=True))


def add_easy_parser(subparsers: Any) -> None:
    easy = subparsers.add_parser("easy", help="one-line data / observation / model / train / infer / eval / run workflows")
    sub = easy.add_subparsers(dest="easy_command", required=True)

    p = sub.add_parser("data", help="generate complete records (numerical solver) or inspect existing ones")
    p.add_argument("--source", choices=("generate", "local", "manifest"), default="generate")
    p.add_argument("--pde")
    p.add_argument("--boundary", default="periodic")
    p.add_argument("--setting", default="smooth_grf")
    p.add_argument("--regime", default="low")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--frames", type=int, default=None, help="stored time frames for temporal PDEs (>=2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", default="hdf5")
    p.add_argument("--shard-size", type=int, default=None)
    p.add_argument("--numeric", action="append", default=[], help="numerical option key=value (passed to the existing generator)")
    p.add_argument("--path", help="existing records (local/manifest source)")
    p.add_argument("--manifest", help="identity manifest (manifest source)")
    p.add_argument("--out", help="output directory (generate)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_data)

    p = sub.add_parser("train", help="train a model on records with an explicit budget")
    p.add_argument("--data", required=True)
    p.add_argument("--task", required=True, choices=("recovery", "forward", "inverse", "rollout"))
    p.add_argument("--model", required=True)
    p.add_argument("--preset", choices=("smoke", "paper"))
    p.add_argument("--model-param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--obs", required=True, help="general protocol (random/grid/block/line/horizontal/vertical/boundary/clustered/full), "
                                                "'paper:<view>' or 'custom:<factory>'")
    p.add_argument("--obs-param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--split", default="all", help="all | paper | stored | holdout:<fraction>")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--horizon", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--set", dest="training", action="append", default=[], metavar="KEY=VALUE", help="TrainingConfig override")
    p.add_argument("--label")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_train)

    p = sub.add_parser("infer", help="predict from observations only (no target)")
    p.add_argument("--model-from", required=True, help="model artifact directory (model.json)")
    p.add_argument("--input", help="observation package (.npz) with observations+mask(+geometry)")
    p.add_argument("--data", help="complete records for benchmark-mode inputs (requires --obs)")
    p.add_argument("--obs", default="stored")
    p.add_argument("--obs-param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--horizon", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--free-geometry", action="store_true", help="declare an obstacle-free domain when the input has no geometry")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--out", required=True, help="predictions file (.npz or .h5)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_infer)

    p = sub.add_parser("eval", help="strict-score predictions against independent targets, or evaluate a model on records")
    p.add_argument("--predictions", help="prediction bundle (.npz/.h5) from `easy infer`")
    p.add_argument("--targets", help="targets .npz (key 'target' [+ 'target_ids', 'target_time_indices'])")
    p.add_argument("--model-from", help="records mode: model artifact")
    p.add_argument("--data", help="records mode: complete records")
    p.add_argument("--obs", help="records mode: observation (general or paper:<view>, or 'paper' for all nine)")
    p.add_argument("--obs-param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_eval)

    p = sub.add_parser("run", help="run a whole pipeline config (YAML/JSON)")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_run)

    p = sub.add_parser("validate", help="validate a pipeline config without running")
    p.add_argument("--config", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_validate)

    p = sub.add_parser("plan", help="show what a pipeline config would do")
    p.add_argument("--config", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_plan)

    p = sub.add_parser("list", help="list models / observations / pdes / presets")
    p.add_argument("kind", choices=("models", "observations", "pdes", "presets"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_list)

    p = sub.add_parser("describe", help="describe a model's parameters, constraints and presets")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_describe)


def _obs(args: Any) -> Any:
    from . import make_observation
    return make_observation(args.obs, **_kv(args.obs_param))


def _cmd_data(args: Any) -> int:
    from . import create_dataset, load_dataset
    if args.source == "generate":
        if not args.pde or not args.out:
            raise SystemExit("easy data --source generate needs --pde and --out")
        handle = create_dataset(pde=args.pde, out=args.out, boundary=args.boundary, setting=args.setting, regime=args.regime,
                                num_samples=args.samples, resolution=args.resolution, seed=args.seed, time_steps=args.frames,
                                format=args.format, shard_size=args.shard_size, numerics=_kv(args.numeric) or None)
    else:
        if not args.path:
            raise SystemExit("easy data --source local/manifest needs --path")
        handle = load_dataset(args.path, source=args.source, manifest=args.manifest)
    _emit(handle.to_dict(), args.json)
    return 0


def _cmd_train(args: Any) -> int:
    from . import train
    budget = {"epochs": args.epochs} if args.epochs else ({"max_steps": args.max_steps} if args.max_steps else None)
    result = train(args.data, task=args.task, model=args.model, preset=args.preset, model_params=_kv(args.model_param),
                   observation=_obs(args), out=args.out, budget=budget, seed=args.seed, device=args.device,
                   batch_size=args.batch_size, split=args.split, history_steps=args.history, horizon=args.horizon,
                   training=_kv(args.training) or None, num_workers=args.num_workers, label=args.label, max_samples=args.max_samples)
    _emit({"result": result.to_dict(), "resolved": result.resolved}, args.json)
    return 0


def _cmd_infer(args: Any) -> int:
    from . import InferenceInput, load_predictor, predict
    predictor = load_predictor(args.model_from, device=args.device)
    if args.input:
        if args.obs != "stored":
            raise SystemExit("an observation package carries its own mask: --obs must stay 'stored'")
        bundle = predict(predictor, InferenceInput.load(args.input), observation="stored", horizon=args.horizon,
                         batch_size=args.batch_size, out=args.out, free_geometry=args.free_geometry)
    elif args.data:
        if args.obs == "stored":
            raise SystemExit("benchmark-mode inputs need an explicit --obs protocol")
        bundle = predict(predictor, args.data, observation=_obs(args), horizon=args.horizon, batch_size=args.batch_size,
                         out=args.out, max_samples=args.max_samples)
    else:
        raise SystemExit("give --input (observation package) or --data (records)")
    _emit(bundle.to_dict(), args.json)
    return 0


def _cmd_eval(args: Any) -> int:
    import numpy as np
    from . import evaluate, evaluate_records, evaluate_views, load_predictor
    if args.predictions:
        if not args.targets:
            raise SystemExit("scoring predictions requires --targets (independent target field)")
        with np.load(args.targets, allow_pickle=False) as data:
            targets = data["target"]
            ids = [str(s) for s in data["target_ids"]] if "target_ids" in data.files else None
            times = data["target_time_indices"].tolist() if "target_time_indices" in data.files else None
        report = evaluate(args.predictions, targets, target_ids=ids, target_time_indices=times, out=args.out)
    elif args.model_from and args.data and args.obs:
        predictor = load_predictor(args.model_from, device=args.device)
        if args.obs in {"paper", "paper"}:
            report = evaluate_views(predictor, args.data, out=args.out, batch_size=args.batch_size, device=args.device)
        else:
            report = evaluate_records(predictor, args.data, _obs(args), out=args.out, batch_size=args.batch_size, device=args.device,
                                      max_samples=args.max_samples)
    else:
        raise SystemExit("use --predictions/--targets, or --model-from/--data/--obs")
    _emit(report, args.json)
    status = report.get("status", "valid" if report.get("all_valid") else "invalid")
    return 0 if status == "valid" else 2


def _cmd_run(args: Any) -> int:
    from . import run
    receipt = run(args.config, dry_run=args.dry_run)
    _emit(receipt, args.json)
    return 0 if receipt.get("status") in {"complete", "validated"} else 2


def _cmd_validate(args: Any) -> int:
    from . import validate_pipeline
    _emit(validate_pipeline(args.config), args.json)
    return 0


def _cmd_plan(args: Any) -> int:
    from . import plan
    result = plan(args.config)
    _emit(result["resolved"] if args.json else result["text"], args.json)
    return 0


def _cmd_list(args: Any) -> int:
    from . import describe_observations, list_models, list_pdes, list_presets
    table = {"models": list_models, "observations": describe_observations, "pdes": list_pdes, "presets": list_presets}
    _emit(table[args.kind](), args.json)
    return 0


def _cmd_describe(args: Any) -> int:
    from . import describe_model
    _emit(describe_model(args.model), args.json)
    return 0

"""Strict expansion and execution helpers for benchmark training campaigns."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .masks import MASK_PROTOCOL_NAMES

TRAINING_CAMPAIGN_SCHEMA = "pdeobs.training-campaign/v1"
PAPER_SETTING_CAMPAIGN_SCHEMA = "pdeobs.training-campaign/v2"
PAPER_SETTING_REGISTRY_SCHEMA = "pdeobs.paper-settings/v1"


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if not text or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in text
    ):
        raise ValueError(f"campaign identifier is not path-safe: {value!r}")
    return text


def build_training_plan(
    campaign_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Expand a reviewed Cartesian matrix into unique, explicit JSON rows."""

    source = Path(campaign_path).resolve()
    campaign = load_config(source)
    schema_version = campaign.get("schema_version")
    supported_schemas = {TRAINING_CAMPAIGN_SCHEMA, PAPER_SETTING_CAMPAIGN_SCHEMA}
    if schema_version not in supported_schemas:
        raise ValueError(f"campaign schema must be one of {sorted(supported_schemas)!r}")
    root = Path(repository_root).resolve() if repository_root else source.parents[2]
    paper_setting_campaign = schema_version == PAPER_SETTING_CAMPAIGN_SCHEMA
    setting_registry_path: Path | None = None
    setting_registry_sha: str | None = None
    setting_records: Mapping[str, Any] = {}
    if paper_setting_campaign:
        setting_registry_path = (root / str(campaign.get("model_setting_registry", ""))).resolve()
        if root not in setting_registry_path.parents or not setting_registry_path.is_file():
            raise ValueError("v2 campaigns require model_setting_registry inside the repository")
        registry = load_config(setting_registry_path)
        if registry.get("schema_version") != PAPER_SETTING_REGISTRY_SCHEMA:
            raise ValueError(
                f"model setting registry schema must be {PAPER_SETTING_REGISTRY_SCHEMA!r}"
            )
        raw_records = registry.get("settings")
        if not isinstance(raw_records, Mapping):
            raise ValueError("model setting registry must contain a settings mapping")
        setting_records = raw_records
        setting_registry_sha = sha256_file(setting_registry_path)
    name = _slug(campaign.get("name"))
    task = _slug(campaign.get("task", "recovery"))
    if task != "recovery":
        raise ValueError("the core pilot currently supports recovery only")
    dataset = dict(campaign.get("dataset", {}))
    summary_sha = str(dataset.get("summary_sha256", ""))
    if len(summary_sha) != 64 or any(
        character not in "0123456789abcdef" for character in summary_sha
    ):
        raise ValueError("dataset.summary_sha256 must be a lowercase SHA256 digest")
    matrix = dict(campaign.get("matrix", {}))
    pdes = tuple(_slug(value) for value in matrix.get("pdes", ()))
    masks = tuple(_slug(value) for value in matrix.get("observation_protocols", ()))
    seeds = tuple(int(value) for value in matrix.get("seeds", ()))
    methods = tuple(dict(value) for value in matrix.get("methods", ()))
    if not pdes or not masks or not seeds or not methods:
        raise ValueError("campaign matrix dimensions must be non-empty")
    unknown_masks = set(masks) - set(MASK_PROTOCOL_NAMES)
    if unknown_masks:
        raise ValueError(f"unknown observation protocols: {sorted(unknown_masks)}")
    if (
        len(pdes) != len(set(pdes))
        or len(masks) != len(set(masks))
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("campaign matrix dimensions may not contain duplicates")

    method_rows: list[tuple[str, str, str, str, tuple[str, ...], dict[str, Any] | None]] = []
    for method in methods:
        method_id = _slug(method.get("id"))
        mode = _slug(method.get("mode", "train_eval"))
        if mode not in {"train_eval", "eval"}:
            raise ValueError(f"method {method_id!r} has unsupported mode {mode!r}")
        config = (root / str(method.get("config", ""))).resolve()
        if root not in config.parents or not config.is_file():
            raise ValueError(f"method {method_id!r} config is missing or outside the repository")
        raw_overrides = method.get("overrides", ())
        if isinstance(raw_overrides, (str, bytes)) or not isinstance(raw_overrides, Sequence):
            raise ValueError(f"method {method_id!r} overrides must be a sequence")
        method_overrides = tuple(str(value).strip() for value in raw_overrides)
        if any(not value or "\n" in value or "\r" in value for value in method_overrides):
            raise ValueError(f"method {method_id!r} has an invalid override")
        setting_provenance: dict[str, Any] | None = None
        if paper_setting_campaign:
            resolved_config = yaml.safe_load(config.read_text(encoding="utf-8"))
            if not isinstance(resolved_config, Mapping):
                raise ValueError(f"method {method_id!r} config must contain a mapping")
            setting_id = _slug(method.get("setting_id"))
            raw_record = setting_records.get(setting_id)
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"method {method_id!r} setting_id {setting_id!r} is not in the registry"
                )
            record = dict(raw_record)
            if _slug(record.get("method")) != method_id:
                raise ValueError(
                    f"setting {setting_id!r} belongs to method {record.get('method')!r}, "
                    f"not {method_id!r}"
                )
            if record.get("status") != "verified":
                raise ValueError(
                    f"setting {setting_id!r} is {record.get('status')!r}; only verified "
                    "paper/official settings may enter a v2 campaign"
                )
            if record.get("local_config_sha256") != sha256_file(config):
                raise ValueError(
                    f"setting {setting_id!r} does not attest the exact local config SHA256"
                )
            for field in ("paper_url", "official_code_url"):
                value = str(record.get(field, ""))
                if not value.startswith("https://"):
                    raise ValueError(f"verified setting {setting_id!r} needs an HTTPS {field}")
            revision = str(record.get("official_code_revision", ""))
            if len(revision) != 40 or any(
                character not in "0123456789abcdef" for character in revision
            ):
                raise ValueError(
                    f"verified setting {setting_id!r} needs a lowercase 40-character "
                    "official_code_revision"
                )
            source_locator = str(record.get("source_setting_locator", "")).strip()
            if not source_locator:
                raise ValueError(f"verified setting {setting_id!r} needs source_setting_locator")
            source_file = str(record.get("official_source_file", "")).strip()
            source_sha = str(record.get("official_source_sha256", ""))
            if not source_file or len(source_sha) != 64 or any(
                character not in "0123456789abcdef" for character in source_sha
            ):
                raise ValueError(
                    f"verified setting {setting_id!r} needs an official source file and SHA256"
                )
            config_method = resolved_config.get("method")
            if not isinstance(config_method, Mapping):
                raise ValueError(
                    f"verified setting {setting_id!r} config needs a method mapping"
                )
            if _slug(config_method.get("name")) != method_id:
                raise ValueError(
                    f"verified setting {setting_id!r} config method does not match {method_id!r}"
                )
            config_kwargs = config_method.get("kwargs")
            if not isinstance(config_kwargs, Mapping):
                raise ValueError(
                    f"verified setting {setting_id!r} config needs method.kwargs provenance"
                )
            if config_kwargs.get("upstream_revision") != revision:
                raise ValueError(
                    f"verified setting {setting_id!r} config upstream revision differs "
                    "from the registry"
                )
            if config_kwargs.get("source_file_sha256") != source_sha:
                raise ValueError(
                    f"verified setting {setting_id!r} config source SHA256 differs "
                    "from the registry"
                )
            if record.get("local_config") != config.relative_to(root).as_posix():
                raise ValueError(
                    f"verified setting {setting_id!r} local_config does not name its exact config"
                )
            scope = str(record.get("reproduction_scope", ""))
            if scope not in {
                "exact_benchmark_reproduction",
                "paper_architecture_training_with_declared_task_adapter",
            }:
                raise ValueError(f"verified setting {setting_id!r} has an unsupported scope")
            adapter = record.get("task_adapter")
            if scope == "paper_architecture_training_with_declared_task_adapter":
                if not isinstance(adapter, Mapping):
                    raise ValueError(f"adapted setting {setting_id!r} needs task_adapter")
                if adapter.get("official_architecture_unchanged") is not True:
                    raise ValueError(
                        f"adapted setting {setting_id!r} must preserve the official architecture"
                    )
                if adapter.get("full_observation_control") != "required":
                    raise ValueError(
                        f"adapted setting {setting_id!r} must require full-observation control"
                    )
                for field in (
                    "input_policy",
                    "spatial_adapter",
                    "normalization",
                    "claim_limit",
                ):
                    if not str(adapter.get(field, "")).strip():
                        raise ValueError(
                            f"adapted setting {setting_id!r} needs task_adapter.{field}"
                        )
            supported_pdes = record.get("supported_pdes")
            if (
                isinstance(supported_pdes, (str, bytes))
                or not isinstance(supported_pdes, Sequence)
                or not supported_pdes
            ):
                raise ValueError(f"verified setting {setting_id!r} needs supported_pdes")
            assert setting_registry_path is not None and setting_registry_sha is not None
            setting_provenance = {
                "setting_id": setting_id,
                "registry": setting_registry_path.relative_to(root).as_posix(),
                "registry_sha256": setting_registry_sha,
                **record,
            }
        method_rows.append(
            (
                method_id,
                mode,
                config.relative_to(root).as_posix(),
                sha256_file(config),
                method_overrides,
                setting_provenance,
            )
        )
    if len({item[0] for item in method_rows}) != len(method_rows):
        raise ValueError("method identifiers may not contain duplicates")

    campaign_sha = sha256_file(source)
    rows: list[dict[str, Any]] = []
    run_paths: set[str] = set()
    for index, (
        (method_id, mode, config, config_sha, method_overrides, setting_provenance),
        pde,
        mask,
        seed,
    ) in enumerate(itertools.product(method_rows, pdes, masks, seeds)):
        if paper_setting_campaign:
            assert setting_provenance is not None
            supported = {_slug(value) for value in setting_provenance["supported_pdes"]}
            if pde not in supported:
                raise ValueError(
                    f"setting {setting_provenance['setting_id']!r} is not verified for PDE {pde!r}"
                )
        run_relpath = f"{name}/main/{method_id}/{pde}/{mask}/seed_{seed}"
        if run_relpath in run_paths:
            raise ValueError(f"duplicate campaign output: {run_relpath}")
        run_paths.add(run_relpath)
        run_name = f"{name}-{method_id}-{pde}-{mask}-seed-{seed}"
        row = {
            "schema_version": TRAINING_CAMPAIGN_SCHEMA,
            "index": index,
            "campaign": name,
            "campaign_file": source.relative_to(root).as_posix(),
            "campaign_sha256": campaign_sha,
            "task": task,
            "method": method_id,
            "mode": mode,
            "pde": pde,
            "observation_protocol": mask,
            "seed": seed,
            "config": config,
            "config_sha256": config_sha,
            "run_relpath": run_relpath,
            "dataset": dataset,
            "artifacts": dict(campaign.get("artifacts", {})),
            "smoke": dict(campaign.get("smoke", {})),
            "overrides": [
                f"name={json.dumps(run_name)}",
                f"seed={seed}",
                # The canonical dataset is partitioned by PDE at the
                # top level. Restrict discovery before metadata filtering
                # so independent rows do not scan all 3,360 shards.
                f"data.train_glob={json.dumps(f'{pde}/**/*.h5')}",
                f"data.filters.pde={json.dumps(pde)}",
                f"data.mask.protocol={json.dumps(mask)}",
                *method_overrides,
            ],
        }
        if paper_setting_campaign:
            row["schema_version"] = PAPER_SETTING_CAMPAIGN_SCHEMA
            row["model_setting"] = setting_provenance
        rows.append(row)
    for expected, row in enumerate(rows):
        if row["index"] != expected:
            raise RuntimeError("campaign indices are not contiguous")
    return rows


def write_training_plan(rows: Sequence[Mapping[str, Any]], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    return path


def read_training_plan_row(plan_path: str | Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise IndexError("plan index must be non-negative")
    with Path(plan_path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                row = json.loads(line)
                if row.get("index") != index:
                    raise ValueError(f"plan row {current} stores index {row.get('index')!r}")
                return row
    raise IndexError(f"plan has no row {index}")


def verify_model_setting_attestation(
    repository_root: str | Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a formal plan's exact paper/official-setting registry record."""

    if row.get("schema_version") != PAPER_SETTING_CAMPAIGN_SCHEMA:
        return {
            "status": "not_attested",
            "formal_paper_setting": False,
            "reason": (
                "legacy v1 campaign rows may be used for compact pilot diagnostics only; "
                "they are not paper-setting comparison rows"
            ),
        }
    root = Path(repository_root).resolve()
    raw_setting = row.get("model_setting")
    if not isinstance(raw_setting, Mapping):
        raise ValueError("v2 plan row is missing model_setting provenance")
    setting = dict(raw_setting)
    registry_path = (root / str(setting.get("registry", ""))).resolve()
    if root not in registry_path.parents or not registry_path.is_file():
        raise ValueError("model setting registry is missing or outside the repository")
    actual_sha = sha256_file(registry_path)
    if actual_sha != setting.get("registry_sha256"):
        raise ValueError("model setting registry SHA256 changed after plan materialization")
    registry = load_config(registry_path)
    if registry.get("schema_version") != PAPER_SETTING_REGISTRY_SCHEMA:
        raise ValueError("model setting registry schema changed after plan materialization")
    setting_id = str(setting.get("setting_id", ""))
    raw_records = registry.get("settings")
    if not isinstance(raw_records, Mapping):
        raise ValueError("model setting registry no longer contains a settings mapping")
    raw_record = raw_records.get(setting_id)
    if not isinstance(raw_record, Mapping):
        raise ValueError(f"model setting {setting_id!r} disappeared from its registry")
    embedded_record = {
        key: value
        for key, value in setting.items()
        if key not in {"setting_id", "registry", "registry_sha256"}
    }
    if dict(raw_record) != embedded_record:
        raise ValueError("embedded model setting differs from the attested registry record")
    if setting.get("status") != "verified" or not bool(
        setting.get("local_config_sha256") == row.get("config_sha256")
    ):
        raise ValueError("v2 model setting is no longer verified for this exact config")
    return {
        "status": "verified",
        "formal_paper_setting": True,
        "reproduction_scope": setting["reproduction_scope"],
        "exact_paper_benchmark_reproduction": (
            setting["reproduction_scope"] == "exact_benchmark_reproduction"
        ),
        "task_adapter": setting.get("task_adapter"),
        "setting_id": setting_id,
        "registry": str(registry_path),
        "registry_sha256": actual_sha,
        "paper_url": setting["paper_url"],
        "official_code_url": setting["official_code_url"],
        "official_code_revision": setting["official_code_revision"],
        "source_setting_locator": setting["source_setting_locator"],
        "official_source_file": setting["official_source_file"],
        "official_source_sha256": setting["official_source_sha256"],
        "local_config_sha256": setting["local_config_sha256"],
    }


def verify_dataset_attestation(dataset_root: str | Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the exact final-QC summary without reparsing its 320 MB payload."""

    root = Path(dataset_root).resolve()
    expected = dict(row["dataset"])
    summary = root / str(expected.get("summary_file", "summary.json"))
    if not summary.is_file():
        raise FileNotFoundError(f"dataset summary does not exist: {summary}")
    actual_sha = sha256_file(summary)
    if actual_sha != expected.get("summary_sha256"):
        raise ValueError(
            "dataset summary SHA256 mismatch: "
            f"expected {expected.get('summary_sha256')}, found {actual_sha}"
        )
    return {
        "root": str(root),
        "summary": str(summary),
        "summary_sha256": actual_sha,
        "expected_shards": int(expected["expected_shards"]),
        "expected_samples": int(expected["expected_samples"]),
        "strict_max_pde_loss": float(expected["strict_max_pde_loss"]),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite_values(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_values(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _observation_diagnostic_summary(
    matched: Mapping[str, Any],
    full: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Interpret the full-observation control without inventing a quality cutoff."""

    metric = "relative_l2"
    matched_metrics = matched.get("metrics")
    full_metrics = full.get("metrics")
    matched_value = matched_metrics.get(metric) if isinstance(matched_metrics, Mapping) else None
    full_value = full_metrics.get(metric) if isinstance(full_metrics, Mapping) else None
    raw_threshold = artifacts.get("full_observation_relative_l2_good_max")
    summary: dict[str, Any] = {
        "role": "diagnostic_only",
        "official_result": "matched_observation",
        "metric": metric,
        "matched_value": matched_value,
        "full_observation_value": full_value,
        "good_max": raw_threshold,
        "rule": (
            "When a preregistered good_max accepts full observation but rejects the matched "
            "partial observation, classify the result as observation-limited and do not retry "
            "solely because the partial-observation score is poor."
        ),
    }
    if raw_threshold is None:
        summary.update(
            classification="unclassified_no_preregistered_threshold",
            action="report_both_metrics_without_inventing_a_cutoff",
        )
        return summary
    threshold = float(raw_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("full_observation_relative_l2_good_max must be finite and non-negative")
    if not isinstance(matched_value, (int, float)) or not isinstance(full_value, (int, float)):
        summary.update(
            classification="unclassified_missing_metric",
            action="diagnose_the_evaluation_artifacts",
        )
    elif float(full_value) <= threshold < float(matched_value):
        summary.update(
            classification="observation_limited",
            action="do_not_retry_for_partial_observation_score_alone",
        )
    elif float(full_value) > threshold:
        summary.update(
            classification="model_or_setting_investigation_required",
            action="inspect_the_paper_setting_checkpoint_and_full_observation_artifacts",
        )
    else:
        summary.update(
            classification="matched_and_full_acceptable",
            action="retain_the_official_matched_observation_result",
        )
    return summary


def execute_training_plan_row(
    plan_path: str | Path,
    index: int,
    *,
    dataset_root: str | Path,
    runs_root: str | Path,
    repository_root: str | Path | None = None,
    smoke: bool = False,
    resume_incomplete: bool = False,
    stale_lock_job_id: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and export examples for exactly one owned plan row.

    Continuation is deliberately explicit.  The caller must first prove that
    the prior Slurm owner is terminal, then pass its exact job id.  A different
    live owner, an unattested run directory, or a partial post-training artifact
    fails closed instead of starting duplicate work.
    """

    from .runner import run_evaluate, run_infer, run_train

    repo = Path(repository_root or Path.cwd()).resolve()
    row = read_training_plan_row(plan_path, index)
    config_path = (repo / row["config"]).resolve()
    if repo not in config_path.parents or sha256_file(config_path) != row["config_sha256"]:
        raise ValueError(
            "experiment config is missing, outside the checkout, or changed after planning"
        )
    model_setting_attestation = verify_model_setting_attestation(repo, row)
    attestation = verify_dataset_attestation(dataset_root, row)
    relative = Path(row["run_relpath"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("plan run path must be a safe relative path")
    if smoke:
        relative = Path(row["campaign"]) / "smoke" / Path(*relative.parts[2:])
    run_dir = (Path(runs_root).resolve() / relative).resolve()
    if Path(runs_root).resolve() not in run_dir.parents:
        raise ValueError("resolved run path escaped the run root")
    complete = run_dir / "status.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            return {**payload, "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "row_receipt.json"
    receipt = {
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": sha256_file(plan_path),
        "index": index,
        "row": row,
    }
    existing = [path for path in run_dir.iterdir() if path.name != ".active.lock"]
    if existing and not resume_incomplete:
        raise FileExistsError(
            f"incomplete output already exists at {run_dir}; diagnose it before an explicit retry"
        )
    if resume_incomplete:
        if not receipt_path.is_file():
            raise ValueError("incomplete continuation lacks an immutable row receipt")
        stored_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if stored_receipt != receipt:
            raise ValueError("incomplete continuation receipt differs from this exact plan row")
    else:
        _atomic_json(receipt_path, receipt)
    lock = run_dir / ".active.lock"
    if lock.exists():
        try:
            prior_lock = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing row lock is unreadable; refusing continuation") from exc
        prior_job = str(prior_lock.get("slurm_job_id") or "")
        if not resume_incomplete or not stale_lock_job_id or prior_job != stale_lock_job_id:
            raise FileExistsError(
                "row is already locked; continuation requires its exact verified-terminal Slurm job id"
            )
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(
        descriptor,
        json.dumps(
            {
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "plan": str(Path(plan_path).resolve()),
                "index": index,
            },
            sort_keys=True,
        ).encode(),
    )
    os.close(descriptor)
    overrides = list(row["overrides"])
    mode = str(row.get("mode", "train_eval"))
    if smoke:
        smoke_config = dict(row.get("smoke", {}))
        overrides.extend(
            [
                f"training.epochs={int(smoke_config.get('epochs', 3))}",
                "training.early_stopping_patience=null",
                f"data.max_samples={int(smoke_config.get('max_samples_per_split', 256))}",
                "training.num_workers=0",
                "training.persistent_workers=false",
            ]
        )
        if row.get("schema_version") != PAPER_SETTING_CAMPAIGN_SCHEMA:
            overrides.append(f"method.kwargs.width={int(smoke_config.get('width', 8))}")
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    def stage_result(name: str) -> dict[str, Any] | None:
        marker = run_dir / "_stages" / f"{name}.json"
        if not marker.is_file():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not isinstance(payload.get("result"), Mapping):
            raise ValueError(f"stage marker {name!r} is not a complete result")
        result = dict(payload["result"])
        if not _finite_values(result):
            raise FloatingPointError(f"stage marker {name!r} contains non-finite values")
        return result

    def save_stage(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
        materialized = dict(result)
        _atomic_json(
            run_dir / "_stages" / f"{name}.json",
            {"status": "complete", "result": materialized},
        )
        return materialized
    try:
        torch = None
        training = None
        checkpoint = None
        train_losses: list[float] = []
        if mode == "train_eval":
            try:
                import torch as torch_module

                torch = torch_module
                if not torch.cuda.is_available():
                    raise RuntimeError("campaign training requires a CUDA GPU")
                torch.cuda.reset_peak_memory_stats()
            except ImportError as exc:
                raise RuntimeError("campaign training requires PyTorch") from exc

            training = stage_result("train")
            if training is None:
                last_checkpoint = run_dir / "train" / "checkpoints" / "last.pt"
                train_dir = run_dir / "train"
                train_existing = list(train_dir.iterdir()) if train_dir.is_dir() else []
                resume_checkpoint = last_checkpoint if last_checkpoint.is_file() else None
                if train_existing and resume_checkpoint is None:
                    raise FileExistsError(
                        "incomplete training has no last.pt; diagnose it before continuation"
                    )
                stage = time.perf_counter()
                training = run_train(
                    config_path=config_path,
                    overrides=overrides,
                    output=train_dir,
                    resume=resume_checkpoint,
                )
                stage_seconds["train"] = time.perf_counter() - stage
                training = save_stage("train", training)
            checkpoint = Path(training["checkpoint"])
            required = (checkpoint, run_dir / "train" / "checkpoints" / "last.pt")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"training did not create required checkpoints: {missing}")

        evaluation_path = run_dir / "evaluation" / "metrics.json"
        evaluation = stage_result("evaluate")
        if evaluation is None:
            if evaluation_path.exists():
                raise FileExistsError("matched evaluation exists without a complete stage marker")
            stage = time.perf_counter()
            evaluation = run_evaluate(
                config_path=config_path,
                overrides=overrides,
                output=evaluation_path,
                checkpoint=checkpoint,
            )
            stage_seconds["evaluate"] = time.perf_counter() - stage
            evaluation = save_stage("evaluate", evaluation)
        if not _finite_values(evaluation):
            raise FloatingPointError("evaluation contains a non-finite numeric value")

        # Full observation is a separate inference-time control. It uses the
        # same trained checkpoint and test split, never overwrites the official
        # matched-mask report, and never triggers duplicate training.
        full_overrides = [
            *overrides,
            'evaluation.observation_mode="full"',
            "evaluation.mask_protocols=[]",
        ]
        full_evaluation_path = run_dir / "evaluation" / "full_observation" / "metrics.json"
        full_observation_evaluation = stage_result("evaluate_full_observation")
        if full_observation_evaluation is None:
            if full_evaluation_path.exists():
                raise FileExistsError(
                    "full-observation evaluation exists without a complete stage marker"
                )
            stage = time.perf_counter()
            full_observation_evaluation = run_evaluate(
                config_path=config_path,
                overrides=full_overrides,
                output=full_evaluation_path,
                checkpoint=checkpoint,
            )
            stage_seconds["evaluate_full_observation"] = time.perf_counter() - stage
            full_observation_evaluation = save_stage(
                "evaluate_full_observation", full_observation_evaluation
            )
        if not _finite_values(full_observation_evaluation):
            raise FloatingPointError(
                "full-observation evaluation contains a non-finite numeric value"
            )

        example_count = int(dict(row.get("artifacts", {})).get("prediction_example_samples", 64))
        inference_path = run_dir / "prediction_examples.h5"
        inference = stage_result("prediction_examples")
        if inference is None:
            if inference_path.exists():
                raise FileExistsError("prediction HDF5 exists without a complete stage marker")
            stage = time.perf_counter()
            inference = run_infer(
                config_path=config_path,
                overrides=[*overrides, f"data.max_samples={example_count}"],
                output=inference_path,
                checkpoint=checkpoint,
            )
            stage_seconds["prediction_examples"] = time.perf_counter() - stage
            inference = save_stage("prediction_examples", inference)

        full_inference_path = run_dir / "full_observation" / "prediction_examples.h5"
        full_observation_inference = stage_result("prediction_examples_full_observation")
        if full_observation_inference is None:
            if full_inference_path.exists():
                raise FileExistsError(
                    "full-observation prediction HDF5 exists without a complete stage marker"
                )
            stage = time.perf_counter()
            full_observation_inference = run_infer(
                config_path=config_path,
                overrides=[
                    *full_overrides,
                    f"data.max_samples={example_count}",
                ],
                output=full_inference_path,
                checkpoint=checkpoint,
            )
            stage_seconds["prediction_examples_full_observation"] = (
                time.perf_counter() - stage
            )
            full_observation_inference = save_stage(
                "prediction_examples_full_observation", full_observation_inference
            )

        if mode == "train_eval":
            history = json.loads((run_dir / "train" / "history.json").read_text(encoding="utf-8"))
            train_losses = [float(item["train_loss"]) for item in history]
            if not train_losses or not all(math.isfinite(value) for value in train_losses):
                raise FloatingPointError("training history is empty or non-finite")
            if smoke and bool(
                row.get("smoke", {}).get("require_nonincreasing_training_loss", True)
            ):
                if min(train_losses[1:] or train_losses) > train_losses[0]:
                    raise RuntimeError("smoke training loss never improved after the first epoch")

        gpu_memory = None
        if torch is not None:
            gpu_memory = {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            }
        runtime = {
            "stage_seconds": stage_seconds,
            "total_seconds": time.perf_counter() - started,
            "gpu_memory": gpu_memory,
        }
        _atomic_json(run_dir / "runtime.json", runtime)
        result = {
            "status": "complete",
            "smoke": smoke,
            "plan_index": index,
            "row": row,
            "dataset_attestation": attestation,
            "model_setting_attestation": model_setting_attestation,
            "training": training,
            "evaluation": evaluation,
            "inference": inference,
            "full_observation_evaluation": full_observation_evaluation,
            "full_observation_inference": full_observation_inference,
            "observation_diagnostic": _observation_diagnostic_summary(
                evaluation,
                full_observation_evaluation,
                dict(row.get("artifacts", {})),
            ),
            "runtime": runtime,
            "training_loss_first": train_losses[0] if train_losses else None,
            "training_loss_best": min(train_losses) if train_losses else None,
        }
        _atomic_json(complete, result)
        return result
    except BaseException as exc:
        failure = {
            "status": "failed",
            "smoke": smoke,
            "plan_index": index,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        failures = run_dir / "_failures"
        failure_name = (
            f"failure-{int(time.time() * 1_000_000)}-"
            f"{os.environ.get('SLURM_JOB_ID', 'local')}.json"
        )
        _atomic_json(failures / failure_name, failure)
        _atomic_json(run_dir / "failure.json", failure)
        raise
    finally:
        lock.unlink(missing_ok=True)

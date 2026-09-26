from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_resolve_device_adds_visible_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import resolve_device

    monkeypatch.setenv("SLURM_LOCALID", "5")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_device("cuda") == torch.device("cuda:1")
    assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_amp_initial_scale_is_fail_closed() -> None:
    from pdeobs.training import TrainingConfig

    assert TrainingConfig().amp_initial_scale == 1.0
    with pytest.raises(ValueError, match="amp_initial_scale"):
        TrainingConfig(amp_initial_scale=0.0)
    with pytest.raises(ValueError, match="amp_initial_scale"):
        TrainingConfig(amp_initial_scale=float("nan"))


def test_loss_spike_scope_is_explicit() -> None:
    from pdeobs.training import TrainingConfig

    assert TrainingConfig().loss_spike_scope == "batch"
    assert TrainingConfig(loss_spike_scope="epoch").loss_spike_scope == "epoch"
    with pytest.raises(ValueError, match="loss_spike_scope"):
        TrainingConfig(loss_spike_scope="stratum")


def test_epoch_loss_spike_scope_does_not_compare_heterogeneous_batches(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig

    def model() -> object:
        layer = torch.nn.Conv2d(1, 1, kernel_size=1)
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        return layer

    baseline = {
        "input": torch.zeros((2, 1, 4, 4)),
        "target": torch.ones((2, 1, 4, 4)),
    }
    harder = {
        "input": torch.zeros((2, 1, 4, 4)),
        "target": torch.full((2, 1, 4, 4), 4.0),
    }
    loader = [baseline] * 20 + [harder] * 3
    shared = dict(
        epochs=1,
        loss="mse",
        learning_rate=0.0,
        amp=False,
        device="cpu",
        health_policy="fail",
        loss_spike_ratio=10.0,
        loss_spike_consecutive_windows=3,
        gradient_norm_warning=1.0e9,
        checkpoint_every=1,
    )

    batch_trainer = Trainer(
        model(),
        TrainingConfig(
            **shared,
            loss_spike_scope="batch",
            checkpoint_dir=str(tmp_path / "batch"),
        ),
    )
    with pytest.raises(RuntimeError, match="sustained_loss_spike"):
        batch_trainer.fit(loader)

    epoch_trainer = Trainer(
        model(),
        TrainingConfig(
            **shared,
            loss_spike_scope="epoch",
            checkpoint_dir=str(tmp_path / "epoch"),
        ),
    )
    history = epoch_trainer.fit(loader)
    assert len(history) == 1
    assert epoch_trainer.health_events == []


def test_relative_losses_reduce_in_float32() -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import _loss_function

    prediction = torch.full((1, 1, 128, 128), 0.25, dtype=torch.float16)
    target = torch.full((1, 1, 128, 128), 0.125, dtype=torch.float16)
    for name in ("relative_l1", "relative_l2"):
        loss = _loss_function(name)(prediction, target)
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)


def test_checkpoint_is_weights_only_safe_and_restores_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.runner import _load_checkpoint
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=1,
            amp=False,
            device="cpu",
            checkpoint_dir=str(tmp_path),
        ),
    )
    checkpoint = trainer.save_checkpoint("safe.pt", epoch=1)

    payload = load_checkpoint_payload(checkpoint)
    assert payload["format_version"] == 2
    assert isinstance(payload["rng"]["numpy"], dict)

    restored = torch.nn.Conv2d(1, 1, kernel_size=1)
    _load_checkpoint(restored, checkpoint)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_step_scheduler_state_and_history_resume_exactly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    loader = [
        {
            "input": torch.ones((2, 1, 4, 4)),
            "target": torch.zeros((2, 1, 4, 4)),
        }
    ]
    first = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(
            epochs=1,
            optimizer="adam",
            learning_rate=0.01,
            weight_decay=0.0,
            loss="relative_l1",
            scheduler="step",
            scheduler_step_size=1,
            scheduler_gamma=0.5,
            amp=False,
            grad_clip=None,
            device="cpu",
            checkpoint_dir=str(tmp_path),
        ),
    )
    first.fit(loader)
    checkpoint = tmp_path / "last.pt"
    payload = load_checkpoint_payload(checkpoint)
    assert payload["scheduler_state"] is not None
    assert payload["history"][0]["learning_rate"] == pytest.approx(0.01)

    resumed = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(
            epochs=2,
            optimizer="adam",
            learning_rate=0.01,
            weight_decay=0.0,
            loss="relative_l1",
            scheduler="step",
            scheduler_step_size=1,
            scheduler_gamma=0.5,
            amp=False,
            grad_clip=None,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            resume_from=str(checkpoint),
        ),
    )
    history = resumed.fit(loader)
    assert resumed.start_epoch == 1
    assert [row["learning_rate"] for row in history] == pytest.approx([0.01, 0.005])


def test_one_cycle_scheduler_steps_per_batch_and_resumes_exactly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    loader = [
        {"input": torch.ones((1, 1, 4, 4)), "target": torch.zeros((1, 1, 4, 4))},
        {"input": torch.ones((1, 1, 4, 4)), "target": torch.zeros((1, 1, 4, 4))},
    ]
    config = TrainingConfig(
        epochs=2,
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=0.0,
        loss="relative_l1",
        scheduler="one_cycle",
        scheduler_steps_per_epoch=2,
        amp=False,
        grad_clip=None,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    first = Trainer(torch.nn.Conv2d(1, 1, kernel_size=1), config)
    first.run_epoch(loader, training=True)
    learning_rate_after_half = float(first.optimizer.param_groups[0]["lr"])
    checkpoint = first.save_checkpoint("one-cycle-half.pt", epoch=1)

    payload = load_checkpoint_payload(checkpoint)
    assert payload["optimizer_steps"] == 2
    assert payload["scheduler_state"]["last_epoch"] == 2
    assert not any(callable(value) for value in payload["scheduler_state"].values())

    resumed = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(**{**vars(config), "resume_from": str(checkpoint)}),
    )
    assert float(resumed.optimizer.param_groups[0]["lr"]) == pytest.approx(
        learning_rate_after_half
    )
    resumed.run_epoch(loader, training=True)
    assert resumed._optimizer_steps == 4
    assert resumed.scheduler.last_epoch == 4


def test_checkpoint_loader_rejects_unsafe_pickle_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import load_checkpoint_payload

    checkpoint = tmp_path / "legacy-unsafe.pt"
    torch.save({"array": np.arange(3)}, checkpoint)

    with pytest.raises(ValueError, match="refuses arbitrary-pickle checkpoints"):
        load_checkpoint_payload(checkpoint)


def test_training_health_records_optimizer_signals_and_survives_checkpoint(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    loader = [
        {
            "input": torch.randn((2, 1, 4, 4)),
            "target": torch.randn((2, 1, 4, 4)),
        }
    ]
    trainer = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(
            epochs=1,
            amp=False,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            health_policy="record",
        ),
    )

    history = trainer.fit(loader)

    assert history[0]["throughput_samples_per_second"] > 0
    assert history[0]["preclip_gradient_norm_max"] > 0
    assert history[0]["postclip_gradient_norm_max"] > 0
    assert history[0]["peak_gpu_memory_bytes"] == 0
    payload = load_checkpoint_payload(tmp_path / "last.pt")
    assert payload["optimizer_steps"] == 1
    assert payload["health_events"] == []


def test_fail_fast_health_policy_stops_a_sustained_large_gradient(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig

    loader = [
        {
            "input": torch.full((2, 1, 4, 4), 1.0e4),
            "target": torch.zeros((2, 1, 4, 4)),
        }
    ]
    trainer = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(
            epochs=1,
            loss="mse",
            amp=False,
            grad_clip=None,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            health_policy="fail",
            gradient_norm_warning=1.0,
            gradient_warning_consecutive_windows=1,
        ),
    )

    with pytest.raises(RuntimeError, match="sustained_large_gradient"):
        trainer.fit(loader)
    assert trainer.health_events[0]["kind"] == "sustained_large_gradient"


def test_source_faithful_clipping_records_raw_gradient_as_nonfatal_warning(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from pdeobs.training import Trainer, TrainingConfig, load_checkpoint_payload

    loader = [
        {
            "input": torch.full((2, 1, 4, 4), 1.0e4),
            "target": torch.zeros((2, 1, 4, 4)),
        }
    ]
    trainer = Trainer(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        TrainingConfig(
            epochs=1,
            loss="mse",
            amp=False,
            grad_clip=0.1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            health_policy="fail",
            gradient_norm_warning=1.0,
            gradient_warning_consecutive_windows=1,
            gradient_warning_fatal=False,
        ),
    )

    history = trainer.fit(loader)

    assert trainer.health_events == []
    assert trainer.health_warnings[0]["kind"] == "sustained_large_gradient"
    assert history[0]["gradient_warning_steps"] == 1
    payload = load_checkpoint_payload(tmp_path / "last.pt")
    assert payload["health_events"] == []
    assert payload["health_warnings"] == trainer.health_warnings

"""Boundary regressions discovered by the independent uploaded-archive audit.

These exercise the API facade, not numerical kernels or empirical benchmark quality.
"""
from __future__ import annotations

import importlib
import json
import numpy as np
import pytest

from pdeobs import api
from pdeobs.api.infer import PredictionBundle


def _bundle():
    targets = np.stack([np.ones((4, 4, 1)), np.full((4, 4, 1), 2)]).astype(np.float32)
    inputs = api.InferenceInput(targets.copy(), np.ones_like(targets), sample_ids=['a', 'b'])
    return PredictionBundle(targets.copy(), ['a', 'b'], [0], 'recovery', inputs=inputs), targets


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1.0, 0.5, 2.0])
def test_input_rejects_invalid_mask_before_conversion(bad):
    bundle, targets = _bundle()
    mask = bundle.inputs.mask.copy()
    mask.flat[0] = bad
    with pytest.raises(ValueError, match='mask'):
        api.InferenceInput(targets, mask, sample_ids=['a', 'b'])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_input_rejects_nonfinite_geometry(bad):
    bundle, targets = _bundle()
    geom = np.zeros_like(targets)
    geom.flat[0] = bad
    with pytest.raises(ValueError, match='geometry'):
        api.InferenceInput(targets, bundle.inputs.mask, geometry=geom, sample_ids=['a', 'b'])


def test_correct_target_reordering_is_accepted_and_preserves_score():
    bundle, targets = _bundle()
    report = api.evaluate(bundle, targets[::-1], target_ids=['b', 'a'])
    assert report['status'] == 'valid', report
    assert report['summary']['rel_l2_joint_mean'] == 0.0


def test_numpy_target_ids_are_supported():
    bundle, targets = _bundle()
    report = api.evaluate(bundle, targets, target_ids=np.array(['a', 'b']))
    assert report['status'] == 'valid'


@pytest.mark.parametrize('times', [[0.9], [0.0], [], ['0'], [True]])
def test_invalid_target_times_are_not_silently_coerced(times):
    bundle, targets = _bundle()
    report = api.evaluate(bundle, targets, target_ids=['a', 'b'], target_time_indices=times)
    assert report['status'] == 'invalid'
    assert report['summary'] is None


def test_input_time_indices_are_validated_before_coercion():
    bundle, targets = _bundle()
    with pytest.raises(ValueError, match='time_indices'):
        api.InferenceInput(targets, bundle.inputs.mask, sample_ids=['a', 'b'], time_indices=[0.9])


@pytest.mark.parametrize('bad', [np.nan, 0.5, 2.0])
def test_postconstruction_mutation_of_mask_is_still_rejected(bad):
    bundle, targets = _bundle()
    bundle.inputs.mask.flat[0] = bad
    report = api.evaluate(bundle, targets)
    assert report['status'] == 'invalid'


def test_prediction_bundle_wrong_schema_is_rejected(tmp_path):
    bundle, _ = _bundle()
    path = bundle.save(tmp_path / 'prediction.npz')
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    meta = json.loads(str(values['schema'])); meta['schema_version'] = 'unrelated/v999'
    values['schema'] = np.asarray(json.dumps(meta))
    np.savez(path, **values)
    with pytest.raises(ValueError, match='schema'):
        PredictionBundle.load(path)


def test_prediction_bundle_float_frames_are_rejected(tmp_path):
    bundle, _ = _bundle()
    path = bundle.save(tmp_path / 'prediction.npz')
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values['prediction_time_indices'] = np.array([0.9])
    np.savez(path, **values)
    with pytest.raises(ValueError, match='time_indices'):
        PredictionBundle.load(path)


@pytest.mark.parametrize('stages', [['trian'], [], 'train', ['data', 'data'], ['data', None]])
def test_pipeline_rejects_misspelled_or_malformed_stages(stages):
    cfg = {'stages': stages, 'data': {'source': 'generate', 'pde': 'poisson', 'out': 'unused'}}
    with pytest.raises(ValueError, match='stage'):
        api.validate_pipeline(cfg)


def test_unknown_stage_does_not_create_success_receipt(tmp_path):
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='stage'):
        api.run({'out': str(out), 'stages': ['trian']})
    assert not out.exists()


def test_invalid_score_propagates_to_failed_pipeline_receipt(tmp_path, monkeypatch):
    # Deliberate failure injection: this tests status propagation, not model quality.
    mod = importlib.import_module('pdeobs.api.evaluate')
    monkeypatch.setattr(mod, 'evaluate', lambda *a, **kw: {'status': 'invalid', 'summary': None, 'errors': ['injected strict failure']})
    cfg = {
        'task': 'recovery', 'device': 'cpu', 'out': str(tmp_path / 'run'),
        'data': {'source': 'generate', 'pde': 'poisson', 'out': str(tmp_path / 'data'),
                 'num_samples': 4, 'resolution': 16, 'seed': 7},
        'observation': {'name': 'random', 'ratio': 0.5},
        'model': {'name': 'zero'}, 'train': {'budget': {'max_steps': 1}, 'batch_size': 2},
        'predict': {}, 'evaluate': {},
    }
    with pytest.raises(ValueError, match='strict evaluation'):
        api.run(cfg)
    receipt = json.loads((tmp_path / 'run/pipeline_receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['stages']['evaluate']['status'] == 'invalid'

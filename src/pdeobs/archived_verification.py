"""Inference-only, append-only verification of a pinned archived checkpoint.

This is NOT paper-row training, an acceptance policy, or replacement of the
archived index. It retains raw tensors and applies the unchanged strict scorer.
The temporal coordinate is explicitly stored-frame POSITION; each identity's
source solver indices and physical times are bound separately. Different
physical regimes need not have identical solver timesteps.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import yaml

from .dataset import BenchmarkDataset
from .evaluation import _numpy_channels_last, predict_batch
from .one_setting import stable_split, subset_dataset, VIEW_ORDER
from .runner import _evaluation_config, _load_checkpoint, _method
from .storage import sha256_file
from .strict_inference import _StrictWriter, batches, metadata_target_shape, target_frame_metadata
from .strict_score import CONTRACT_VERSION, _canonical_hash, score_prediction_file, write_report
from .training import unpack_batch_context

VERSION = 'pdeobs-archived-prediction-verification-v1'


def require_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f'input hash mismatch: {path.name}')


def assert_split(actual: dict, expected: dict) -> None:
    """Compare scientific fields; historical schema tag v7 differs from helper v1."""
    keys = ('algorithm', 'seed', 'macrodomain_identity', 'records', 'train_records',
            'validation_records', 'test_records', 'regime_counts',
            'train_sample_ids_sha256', 'test_sample_ids_sha256')
    if any(actual.get(k) != expected.get(k) for k in keys):
        raise ValueError('reconstructed identity split differs from archive')


def compare_score(report: dict, archived: dict) -> dict:
    old = archived['relative_l2']
    result = {'archived_relative_l2': old, 'strict_status': report['status'],
              'strict_relative_l2': None, 'absolute_difference': None,
              'relative_difference': None, 'within_reporting_tolerance': False,
              'tolerance': {'atol': 1e-7, 'rtol': 1e-4},
              'note': 'Tolerance classifies reproducibility, never scientific quality.'}
    if report['status'] == 'valid':
        new = report['summary']['rel_l2_joint_mean']
        result.update(strict_relative_l2=new, absolute_difference=abs(new-old),
                      relative_difference=abs(new-old)/max(abs(old), 1e-12),
                      within_reporting_tolerance=bool(np.isclose(new, old, atol=1e-7, rtol=1e-4)))
    return result


def export_bound_predictions(method: Any, loader: Any, *, config: Any,
                             contract: dict, prediction_path: Path) -> dict:
    """Export every identity, including adverse finite or nonfinite predictions."""
    writer = _StrictWriter(prediction_path, contract)
    writer.handle.attrs['verification_adapter'] = VERSION
    writer.handle.attrs['time_index_semantics'] = contract['time_index_semantics']
    seen = set()
    expected = set(contract['expected_ids'])
    try:
        for batch in loader:
            inputs, masks, _, geometry, metadata = unpack_batch_context(batch)
            ids = [r['sample_id'] for r in metadata]
            if len(set(ids)) != len(ids) or set(ids)-expected or seen.intersection(ids):
                raise ValueError('unexpected/repeated batch identities')
            selected = []
            for row in metadata:
                frames = target_frame_metadata(row, config)
                if frames != contract['frame_mapping_by_identity'][row['sample_id']]:
                    raise ValueError('source time mapping changed during inference')
                if frames['target_stored_positions'] != contract['target_time_indices']:
                    raise ValueError('task target positions differ from contract')
                selected.append(frames)
            prediction, target = predict_batch(method, batch, config)
            if list(target.shape[1:]) != contract['expected_shape']:
                raise ValueError('target shape differs from pinned contract')
            for name, values in (('prediction', prediction), ('target', target),
                                 ('observation', inputs), ('mask', masks), ('geometry', geometry)):
                if values is not None:
                    writer.append(name, _numpy_channels_last(values, 'channels_last'))
            writer.append('prediction_ids', ids, strings=True)
            writer.append('target_ids', ids, strings=True)
            writer.append('metadata_json', [json.dumps({**r, **f}, sort_keys=True, allow_nan=False)
                          for r, f in zip(metadata, selected)], strings=True)
            writer.count += len(ids)
            seen.update(ids)
        if seen != expected:
            raise ValueError('inference omitted expected identities')
        for key in ('prediction_time_indices', 'target_time_indices'):
            writer.handle.create_dataset(key, data=np.asarray(contract['target_time_indices'], dtype=np.int64))
        writer.close(True)
    except BaseException:
        writer.close(False)
        raise
    return score_prediction_file(prediction_path, contract)


def released_digests(released: dict) -> dict:
    """Digests of one released model row.  Manifest v3 lists only the digests of the published files
    (``checkpoint_sha256``, ``files[rel].sha256``); v2 rows also carried the archived digests
    (``checkpoint_original_sha256``, ``files[rel].anonymized_sha256``)."""
    if 'checkpoint_sha256' in released:
        return {'checkpoint': released['checkpoint_sha256'],
                'files': {rel: entry['sha256'] for rel, entry in released['files'].items()},
                'original': released.get('checkpoint_original_sha256')}
    return {'checkpoint': released['checkpoint_anonymized_sha256'],
            'files': {rel: entry['anonymized_sha256'] for rel, entry in released['files'].items()},
            'original': released.get('checkpoint_original_sha256')}


def verify_archived_row(*, row: dict, released: dict, model_root: Path,
                        data_root: Path, data_manifest: dict, campaign: dict,
                        output: Path, device: str = 'cuda:0',
                        release_map: dict | None = None) -> dict:
    """One fixed model, nine views, full 200 identities; zero optimizer updates.

    ``release_map`` is this repository's record for the identity (results/public_deposits/release_map.json):
    it binds the archived checkpoint digest to the published one when the released manifest lists only
    published digests."""
    output.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    completion = {'schema_version': VERSION, 'identity': row['identity'],
                  'status': 'failed', 'started_unix': begin, 'blocks': [],
                  'historical_completion_credit_delta': 0, 'optimizer_updates': 0}
    dataset = None
    try:
        if row['identity'] != released['identity']:
            raise ValueError('release identity differs from archive')
        published = released_digests(released)
        original = published['original'] or (release_map or {}).get('checkpoint_original_sha256')
        if original is None:
            raise ValueError('no archived checkpoint identity for this release row: pass the release map')
        if row['checkpoint_id'] != original:
            raise ValueError('original checkpoint identity differs from archive')
        if release_map and release_map.get('checkpoint_sha256') not in (None, published['checkpoint']):
            raise ValueError('published checkpoint digest differs from the release map')
        checkpoint = model_root/'checkpoints/last.pt'
        require_hash(checkpoint, published['checkpoint'])
        require_hash(model_root/'resolved.yaml', published['files']['resolved.yaml'])
        config = yaml.safe_load((model_root/'resolved.yaml').read_text())
        if config['task'] != row['task']:
            raise ValueError('task differs between resolved configuration and archive')
        if config.get('seed') != row['split']['seed']:
            raise ValueError('mask seed differs from original campaign seed')
        original_mask = campaign['views'][row['train_view']]['mask']
        if config['data']['mask'] != original_mask:
            raise ValueError('training mask differs from frozen observation definition')
        ec = replace(_evaluation_config(config), device=device, data_layout='channels_last')
        options = dict(task=ec.task, target_step=ec.target_step, horizon=ec.horizon,
                       history_steps=ec.history_steps, seed=int(config['seed']),
                       mask=original_mask,
                       state_representation=config['data'].get('state_representation', 'native'))
        paths = [data_root/x['path'] for x in data_manifest['shards']]
        # The controller hashes shards before any GPU work; stable inode metadata
        # is rechecked on each row and at its close. No new data is generated.
        for p, entry in zip(paths, data_manifest['shards']):
            st = p.stat()
            if st.st_size != entry['bytes'] or st.st_mtime_ns != entry['mtime_ns']:
                raise ValueError('preflight-verified dataset file changed')
        dataset = BenchmarkDataset(paths, **options)
        _, positions, split = stable_split(dataset.metadata, row['split']['seed'],
                                           row['split']['macrodomain_identity'])
        assert_split(split, row['split'])
        test = subset_dataset(dataset, positions)
        ids = [m['sample_id'] for m in test.metadata]
        if len(ids) != 200 or len(set(ids)) != 200:
            raise ValueError('not exactly 200 distinct held-out identities')
        frame_map = {m['sample_id']: target_frame_metadata(m, ec) for m in test.metadata}
        shape = metadata_target_shape(test, ec)
        target_positions = [1, 2, 3] if ec.task == 'rollout' else [0]
        if any(f['target_stored_positions'] != target_positions for f in frame_map.values()):
            raise ValueError('not the original stored-frame task')
        if set(row['blocks']) != set(VIEW_ORDER):
            raise ValueError('archive is not the full nine-view block set')
        method = _method(config)
        _load_checkpoint(method, checkpoint, device=device)
        import torch
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        batch_size = int(config['training']['batch_size'])
        write_report(output/'provenance.json', {'schema_version': VERSION,
                    'archived_row': row, 'released_file': released,
                    'dataset_manifest': data_manifest, 'split': split,
                    'original_checkpoint_id': row['checkpoint_id'],
                    'actual_checkpoint_id': published['checkpoint'],
                    'mask_seed':config['seed'], 'inference_batch_size': batch_size,
                    'torch_version':torch.__version__, 'cuda_version':torch.version.cuda,
                    'device_name':torch.cuda.get_device_name() if device.startswith('cuda') else 'cpu',
                    'note':'Verification only; historical training and evaluation remain unchanged.'})
        # Matched view first provides an early sanity check without selecting any
        # model by test score. Every remaining view is still executed exactly once.
        order = [row['train_view']] + [v for v in VIEW_ORDER if v != row['train_view']]
        for view in order:
            block_begin = time.time()
            test.mask_config = dict(campaign['views'][view]['mask'])
            dest = output/'blocks'/view
            dest.mkdir(parents=True, exist_ok=False)
            archived_ec = row['blocks'][view].get('evaluation_config', {})
            for key in ('task','horizon','history_steps','rollout_target_offset','target_step','observation_mode'):
                if key in archived_ec and getattr(ec,key) != archived_ec[key]:
                    raise ValueError('evaluation configuration drift: '+key)
            contract = {'schema_version':CONTRACT_VERSION, 'task':ec.task, 'split':'paper-test',
                        'expected_ids':ids, 'expected_shape':shape, 'target_time_indices':target_positions,
                        'time_index_semantics':'stored_frame_positions; per_identity_source_mapping_bound',
                        'frame_mapping_by_identity':frame_map, 'projection':False,
                        'observation_id':view, 'mask_config':test.mask_config,
                        'checkpoint_id':published['checkpoint'],
                        'original_checkpoint_id':row['checkpoint_id'],
                        'dataset_manifest_sha256':_canonical_hash(data_manifest),
                        'historical_split_sha256':_canonical_hash(split),
                        'training_cohort':row['training_cohort'], 'actual_epochs':row['actual_epochs'],
                        'identity':row['identity'], 'verification_adapter':VERSION}
            write_report(dest/'contract.json', contract)
            report = export_bound_predictions(method, batches(test, batch_size), config=ec,
                        contract=contract, prediction_path=dest/'predictions.h5')
            write_report(dest/'score.json', report)
            comparison = compare_score(report, row['blocks'][view])
            write_report(dest/'comparison.json', comparison)
            item = {'test_view':view, 'status':report['status'],
                    'samples':report['scored_identity_count'], 'seconds':time.time()-block_begin,
                    'predictions_sha256':sha256_file(dest/'predictions.h5'),
                    'score_sha256':sha256_file(dest/'score.json'), 'comparison':comparison}
            write_report(dest/'completion.json', item)
            completion['blocks'].append(item)
            print(json.dumps({'identity':row['identity'], **item}), flush=True)
        for p, entry in zip(paths, data_manifest['shards']):
            st = p.stat()
            if st.st_size != entry['bytes'] or st.st_mtime_ns != entry['mtime_ns']:
                raise ValueError('dataset changed while evaluating')
        completion['status'] = 'complete' if all(b['status']=='valid' for b in completion['blocks']) else 'invalid_predictions'
    except Exception as exc:
        completion['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if dataset is not None:
            dataset.close()
        completion['finished_unix'] = time.time()
        completion['seconds'] = time.time()-begin
        write_report(output/'completion.json', completion)
    return completion

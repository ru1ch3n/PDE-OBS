#!/usr/bin/env python3
"""Small AI entrypoint: read-only doctor/context, explicit bounded CPU smoke.

No package installation, network access, cloud/scheduler interaction or arbitrary shell execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def dump(value, path=None):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n'
    if path is not None:
        Path(path).write_text(text, encoding='utf-8')
    return text


def inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def doctor():
    packages = {}
    for name in ['numpy', 'scipy', 'h5py', 'PyYAML', 'torch', 'pytest', 'pdeobs']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    index = json.loads((ROOT / 'docs/ai/repository_index.json').read_text())
    paths = sorted({p for values in index['tasks'].values() for p in values} | set(index['entrypoints'].values()))
    missing = [p for p in paths if not (ROOT / p).is_file()]
    protected = json.loads((ROOT / 'docs/ai/frozen_science_sha256.json').read_text())['files']
    changed = [p for p, expected in protected.items() if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != expected]
    return {
        'schema_version': 'pdeobs-ai-doctor/v1', 'scope': 'static local check; no package imports/model tests',
        'status': 'needs_attention' if missing or changed else 'static_checks_passed',
        'python': platform.python_version(), 'repository_root': str(ROOT), 'installed_distributions': packages,
        'missing_indexed_files': missing, 'protected_scientific_files_checked': len(protected),
        'changed_protected_files': changed,
        'warnings': [
            'Installed distributions may differ from repository-local source; smoke uses this source explicitly.',
            'Dependency presence does not prove version/platform/GPU compatibility.',
            'No production dataset, checkpoint or scientific-result correctness was checked.',
            'Old acceptance counts overlap; do not add 521, 42, 34, and all-suite totals together.',
        ],
    }


def context(task: str, max_chars: int):
    index = json.loads((ROOT / 'docs/ai/repository_index.json').read_text())
    used = 0
    for relative in index['tasks'][task]:
        path = (ROOT / relative).resolve()
        if not inside(path, ROOT) or path.suffix not in {'.md', '.json', '.yaml', '.py'}:
            raise ValueError('unsafe context path')
        text = path.read_text(encoding='utf-8')
        header = f'\n--- {relative} | sha256={hashlib.sha256(path.read_bytes()).hexdigest()} ---\n'
        numbered = '\n'.join(f'L{i}: {line}' for i, line in enumerate(text.splitlines(), 1)) + '\n'
        remaining = max_chars - used
        if remaining <= len(header):
            print(f'\n[CONTEXT LIMIT; unread file: {relative}]')
            return
        print(header, end=''); used += len(header)
        remaining = max_chars - used
        if len(numbered) > remaining:
            print(numbered[:remaining], end='')
            print(f'\n[TRUNCATED: continue reading {relative} directly; no completeness claim]')
            return
        print(numbered, end=''); used += len(numbered)


def smoke(output: str):
    out = Path(output).expanduser().resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError('choose a fresh output directory; no previous output was removed')
    if inside(out, ROOT) or inside(ROOT, out):
        raise ValueError('output must be outside and not an ancestor of the repository')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from pdeobs import api
    from pdeobs.storage import LazyHDF5Dataset
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {'schema_version': 'pdeobs-ai-smoke/v1', 'status': 'running',
               'scope': 'software acceptance ONLY; 16x16 demo data, NOT paper results',
               'device': 'cpu', 'threads': 1, 'torch': torch.__version__, 'cases': []}
    try:
        for pde, task in [('poisson', 'recovery'), ('heat', 'rollout')]:
            case = out / pde; case.mkdir()
            handle = api.create_dataset(pde=pde, boundary='periodic', setting='smooth_grf', regime='low',
                                        out=case / 'data', num_samples=6, resolution=16, seed=17,
                                        time_steps=4 if task == 'rollout' else None)
            reader = LazyHDF5Dataset(handle.shards)
            try:
                ids = [row['sample_id'] for row in reader.iter_metadata()]
            finally:
                reader.close()
            split = {'train_ids': ids[:4], 'test_ids': ids[4:]}
            obs = api.make_observation('random', ratio=0.5)
            result = api.train(handle, task=task, model='fno', preset='smoke', observation=obs,
                               out=case/'train', budget={'max_steps': 2}, seed=31, device='cpu',
                               batch_size=2, split=split, history_steps=1, horizon=3 if task == 'rollout' else None,
                               num_workers=0)
            package, targets = api.inference_input_from_dataset(handle, obs, task=task, history_steps=1,
                                                               sample_ids=split['test_ids'], seed=0)
            package.save(case/'observations.npz')
            predictor = api.load_predictor(result.artifact.path, device='cpu')
            pred = api.predict(predictor, api.InferenceInput.load(case/'observations.npz'),
                               horizon=3 if task == 'rollout' else None, batch_size=2,
                               out=case/'predictions.npz')
            source_times = handle.stored_frame_indices
            target_times = source_times[1:4] if task == 'rollout' else [source_times[0]]
            np.savez_compressed(case/'targets.npz', target=targets, target_ids=np.asarray(split['test_ids']),
                                target_time_indices=np.asarray(target_times, dtype=np.int64))
            score = api.evaluate(pred, targets, target_ids=split['test_ids'], target_time_indices=target_times,
                                 out=case/'score')
            reloaded = api.predict(api.load_predictor(result.artifact.path, device='cpu'), package,
                                   horizon=3 if task == 'rollout' else None, batch_size=2)
            same = bool(np.array_equal(pred.predictions, reloaded.predictions))
            if score['status'] != 'valid' or not same:
                raise RuntimeError(f'{pde}: strict scoring or fresh-object reload check failed')
            receipt['cases'].append({'pde': pde, 'task': task, 'train_count': 4, 'test_count': 2,
                                     'train_test_overlap': 0, 'optimizer_steps': result.optimizer_steps,
                                     'strict_status': score['status'], 'reload_identical': same,
                                     'targets_used_for_predict': pred.provenance.get('targets_used'),
                                     'note': 'same-process fresh model reload; not independent production reproduction'})
        receipt['status'] = 'passed'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['seconds'] = round(time.perf_counter()-started, 3)
        dump(receipt, out/'ai_smoke_receipt.json')
    print(dump(receipt), end='')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    d = subs.add_parser('doctor', help='read-only metadata and protected-file checks')
    d.add_argument('--json', action='store_true', help='reserved for machine callers; output is always JSON')
    c = subs.add_parser('context', help='allowlisted, line-numbered, bounded source context')
    c.add_argument('--task', choices=['overview','model','scoring','data','paper','evidence'], default='overview')
    c.add_argument('--max-chars', type=int, default=18000)
    s = subs.add_parser('smoke', help='explicitly opt into a fixed tiny CPU workflow')
    s.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        if args.command == 'doctor':
            result = doctor(); print(dump(result), end='')
            return 0 if result['status'] == 'static_checks_passed' else 2
        if args.command == 'context':
            if not 1000 <= args.max_chars <= 100000:
                raise ValueError('max-chars must be between 1000 and 100000')
            context(args.task, args.max_chars)
        else:
            smoke(args.out)
        return 0
    except Exception as exc:
        print(dump({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr, end='')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

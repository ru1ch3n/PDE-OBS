"""Extra test views for one completed, nine-view-evaluated study row (density sweep and held-out mixtures), v2:
waits for a free GPU like the workers do (registers as an EVA waiter so training waiters yield, takes the shared
single-EVA lock), otherwise identical to extra_views_eva.py. Never a paper cell.

  python extra_views_eva_v2.py <eligibility-receipt.json> <output-dir> "<spec>" ["<spec>" ...]
"""
import copy, fcntl, hashlib, json, os, pathlib, subprocess, sys, time, traceback
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent
import mixed_builder                                  # noqa: E402  patches the frozen builder (study view)
import mixed_eva                                      # noqa: E402  load_evaluator500, FIXED500 / SETTINGS_V3
import budget_eva                                     # noqa: E402  settings-v3 evaluator loader
from budget200 import H, J                            # noqa: E402
from pdeobs.mask_specs import parse_mask_spec         # noqa: E402  release grammar (overlay)
BASE = pathlib.Path('/path/to/pdeobs')


def gpu_free(uuid):
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    used = int(subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    return uuid not in apps and not used


def acquire_gpu(gpus, hours=8):
    root = BASE / '_runtime-gates/authorized-the GPU node-evaluation-gpus'
    deadline = time.monotonic() + hours * 3600
    while time.monotonic() < deadline:
        for index, uuid in gpus.items():
            lock = (root / (uuid + '.lock')).open('a+')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); continue
            if not gpu_free(uuid):
                lock.close(); continue
            return lock, int(index), uuid
        time.sleep(10)
    raise RuntimeError('no free GPU within the wait budget')


def resolved_mask(spec_text):
    spec = parse_mask_spec(spec_text)
    if not spec.is_mixture and spec.components[0].protocol != 'full':
        c = spec.components[0]
        return {'protocol': c.protocol, **c.resolved_kwargs((128, 128))}, spec
    return {'protocol': spec.canonical_text()}, spec


def realized_fraction(dataset):
    values = []
    for i in range(len(dataset)):
        md = dataset[i]['metadata']
        if md.get('observation_ratio') is not None:
            values.append(float(md['observation_ratio']))
    return sum(values) / len(values) if values else None


def main():
    plan = J(HERE / 'plan.json'); row = plan['rows'][0]
    receipt = pathlib.Path(sys.argv[1]).resolve(); output = pathlib.Path(sys.argv[2]).resolve(); specs = sys.argv[3:]
    assert specs and not output.exists()
    eligibility = J(receipt); pde, method, view = row['identity'].split('/')
    assert (eligibility['pde'], eligibility['method'], eligibility['training_view']) == (pde, method, view) and eligibility['disposition'] == 'research_accept'
    mixed_eva.verify_overlay(plan)
    module = mixed_eva.load_evaluator500() if row['protocol'] == mixed_eva.FIXED500 else budget_eva.load_evaluator()
    waitroot = BASE / '_runtime-gates/budget200-single-eva-waiters'; waitroot.mkdir(parents=True, exist_ok=True)
    waiter = waitroot / ('extra-' + str(os.getpid()) + '.json')
    with waiter.open('x') as f:
        json.dump(dict(pid=os.getpid(), identity=row['identity'], kind='extra-views'), f)
    try:
        with (BASE / '_runtime-gates/budget200-single-eva.lock').open('a+') as evlock:
            fcntl.flock(evlock, fcntl.LOCK_EX)
            gpu, index, uuid = acquire_gpu(plan['gpus'])
            os.environ['CUDA_VISIBLE_DEVICES'] = uuid
            try:
                import torch
                assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'V100' in torch.cuda.get_device_name(0).upper()
                started = time.time(); output.parent.mkdir(parents=True, exist_ok=True); partial = output.parent / (output.name + '.partial'); partial.mkdir()
                try:
                    root = pathlib.Path(row['output']); cfg = __import__('yaml').safe_load((root / 'resolved.yaml').read_text())
                    campaign, base_config, test_positions, split, checkpoint = module.verify_training_artifacts(
                        repository=pathlib.Path(plan['repository']), campaign_path=pathlib.Path(plan['campaign']), dataset_root=pathlib.Path(plan['dataset']),
                        source_dataset_root=pathlib.Path(cfg['data']['root']), training_output=root, eligibility_path=receipt, eligibility_sha256=H(receipt),
                        pde=pde, method=method, training_view=view)
                    model = module._method(base_config); module._load_checkpoint(model, checkpoint)
                    expected = str(split['test_sample_ids_sha256']); blocks = {}
                    for text in specs:
                        mask, spec = resolved_mask(text); key = spec.view_id
                        view_config = copy.deepcopy(base_config); view_config['name'] = f"{base_config['name']}-extra-{key}"; view_config['data']['mask'] = mask
                        data = module._dataset(view_config, None); assert data is not None and len(data) == 2000
                        view_test = module.subset_dataset(data, test_positions); ids = module.sample_ids(view_test)
                        assert len(ids) == 200 == len(set(ids)) and module.ids_sha256(ids) == expected
                        assert Counter(str(r.get('regime')) for r in view_test.metadata) == Counter({'low': 67, 'medium': 67, 'high': 66})
                        evaluation_config = module._evaluation_config(view_config)
                        context = module._evaluation_context(view_config, view_test, split='test', training_view=view, evaluation_view=key,
                                                             test_sample_ids_sha256=expected, checkpoint_sha256=module.sha256_file(checkpoint),
                                                             **module._evaluation_mask_context(evaluation_config, mask, view_test))
                        result = module.evaluate_model(model, module._loader(view_test, view_config, shuffle=False), config=evaluation_config, context=context)
                        assert result.get('samples') == 200 and module.finite_tree(result)
                        path = partial / 'blocks' / key / 'metrics.json'; module.atomic_json(path, result)
                        blocks[key] = dict(spec=spec.canonical_text(), resolved_mask=mask, samples=200, test_sample_ids_sha256=expected, metrics_sha256=module.sha256_file(path),
                                           relative_l2=result.get('metrics', {}).get('relative_l2'), realized_observation_ratio=realized_fraction(view_test))
                    completion = dict(schema_version='pdeobs.mixed-study-extra-views/v1', status='completed', paper_cell=False, study_id=plan['study_id'], pde=pde, method=method,
                                      training_view=view, training_protocol=row['protocol'], checkpoint_sha256=module.sha256_file(checkpoint), eligibility_sha256=H(receipt),
                                      test_sample_ids_sha256=expected, device=dict(type='cuda', name=torch.cuda.get_device_name(0), uuid=uuid), started_unix=started,
                                      completed_unix=time.time(), blocks=blocks, extra_adapter_sha256=H(__file__))
                    module.atomic_json(partial / 'completion.json', completion); os.replace(partial, output)
                    print(json.dumps({k: v for k, v in completion.items() if k != 'blocks'} | {'relative_l2': {k: b['relative_l2'] for k, b in blocks.items()}}), flush=True)
                except BaseException as exc:
                    module.atomic_json(partial / 'failure.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc())); raise
            finally:
                gpu.close()
    finally:
        waiter.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

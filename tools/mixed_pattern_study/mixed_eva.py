"""Completion-triggered, metadata-first nine-view EVA for the mixed-pattern study rows (the GPU node, no Slurm).
fixed500 rows: the frozen evaluator (base200.py, pinned) with its epoch constant set to 500 and the study view accepted.
settings-v3 rows: same_job_eva.metadata / budget_eva (T2, byte-identical) under the same view patch."""
import hashlib, json, os, sys, time, traceback, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
import mixed_builder                                   # noqa: E402  patches the builder before any evaluator is loaded
import mixed_train                                     # noqa: E402  FIXED500 / SETTINGS_V3
import same_job_eva                                    # noqa: E402  T2: record, POLICY, allocation, metadata (settings-v3)
import budget_eva                                      # noqa: E402  T2: BASE_SHA
from budget200 import H, J, PINS, canonical, replaced  # noqa: E402

FIXED500 = mixed_train.FIXED500; SETTINGS_V3 = mixed_train.SETTINGS_V3
POLICY = same_job_eva.POLICY; record = same_job_eva.record; BASE_SHA = budget_eva.BASE_SHA
STUDY = J(HERE / 'study-authorization.json')
allocation = same_job_eva.allocation  # replaced by the worker: no Slurm allocation exists on the GPU node


def verify_overlay(plan):
    manifest = J(plan['overlay_manifest']); assert H(plan['overlay_manifest']) == plan['overlay_manifest_sha256']
    for rel, item in manifest['files'].items():
        assert H(Path(plan['overlay_src']) / 'pdeobs' / rel) == item['sha256'], rel
    import pdeobs
    assert Path(pdeobs.__file__).resolve().parent == (Path(plan['overlay_src']) / 'pdeobs').resolve()


def load_evaluator500():
    path = HERE / 'base200.py'; assert H(path) == BASE_SHA
    src = path.read_text()
    for old, new in [('completion.get("epochs_completed") != 200', 'completion.get("epochs_completed") != 500'),
                     ('len(records) != 200', 'len(records) != 500'),
                     ('health.get("epochs_completed") != 200', 'health.get("epochs_completed") != 500'),
                     ('identity.get("epochs") != 200', 'identity.get("epochs") != 500')]:
        src = replaced(src, old, new)
    m = types.ModuleType('_mixed_study_eva500'); m.__file__ = str(path)
    exec(compile(src, str(path) + '[fixed500-epochs]', 'exec'), m.__dict__)
    assert m.build_experiment_config is mixed_builder.build_experiment_config
    original = m.atomic_json

    def recorded(path, payload):
        if payload.get('schema_version') == 'pdeobs.one-setting-cross-view-completion/v7':
            payload = dict(payload, training_protocol=FIXED500, training_adapter_sha256=H(HERE / 'mixed_train.py'), actual_training_epochs=500,
                           planned_max_epochs=500, evaluation_metadata_adapter_sha256=H(__file__), frozen_evaluator_sha256=BASE_SHA,
                           study_id=STUDY['study_id'], training_view_kind=mixed_builder.KIND, mask_spec=mixed_builder.SPEC, paper_cell=False)
        return original(path, payload)
    m.atomic_json = recorded
    return m


def metadata500(plan, row, receipt):
    import yaml
    root = Path(row['output']); completion = J(root / 'completion.json'); health = J(root / 'health.json'); history = J(root / 'history.json')
    pde, method, view = row['identity'].split('/')
    assert method != 'jeno' and mixed_builder.accepts(view)
    assert (completion['pde'], completion['method'], completion['training_view']) == (pde, method, view)
    epochs = completion['epochs_completed']; entries = history['history']
    assert epochs == row['epochs'] == 500 and len(entries) == 500
    assert completion['status'] == 'completed' and completion['accepted'] is True
    assert health['status'] == 'completed' and health['accepted'] is True and health['events'] == []
    assert completion['validation_used'] is False and completion['test_records_touched'] == 0
    adapter = H(HERE / 'mixed_train.py')
    for name in ('completion.json', 'health.json', 'history.json', 'identity.json'):
        d = J(root / name)
        assert d['training_protocol'] == FIXED500 and d['study_adapter_sha256'] == adapter and d['study_id'] == STUDY['study_id']
        assert d['mask_spec'] == mixed_builder.SPEC and d['planned_epochs'] == 500 and d['early_stopping'] is False and d['paper_cell'] is False
    assert completion['actual_epochs'] == 500 and completion['stop_reason'] == 'max_epochs_500'
    cfg = yaml.safe_load((root / 'resolved.yaml').read_text())
    proto = J(root / 'study-protocol.json')
    assert proto['protocol'] == FIXED500 and proto['adapter_sha256'] == adapter and proto['source_pins'] == PINS
    assert proto['resolved_config_sha256'] == canonical(cfg) and proto['training_config'] == cfg['training']
    assert cfg['data']['mask'] == proto['mask'] and (mixed_builder.KIND != 'mixture' or proto['mask'] == {'protocol': mixed_builder.SPEC})
    if mixed_builder.KIND == 'seed_subset':
        assert cfg['training']['seed'] == int(mixed_builder.SEED_VIEWS[view]['training_seed'])
    loss = entries[-1]['train_data_loss']
    amendment = __import__('acceptance_policy').classify(row['identity'], completion['checkpoint_sha256'], loss)
    pins = {name: H(root / name) for name in ['completion.json', 'health.json', 'history.json', 'factor_coverage.json', 'split_manifest.json',
                                              'identity.json', 'resolved.yaml', 'provenance.json', 'checkpoints/last.pt',
                                              'checkpoints/training_config.json', 'study-protocol.json']}
    assert pins['checkpoints/last.pt'] == completion['checkpoint_sha256']
    payload = dict(acceptance_policy=POLICY, disposition='research_accept', pde=pde, method=method, training_view=view,
                   checkpoint_sha256=completion['checkpoint_sha256'], epoch=epochs, history_entries=len(entries), training_data_loss_final=loss,
                   **amendment, health_warnings=health.get('warnings'), gradient_warning_steps=sum(x.get('gradient_warning_steps', 0) for x in entries),
                   test_sample_ids_sha256=J(root / 'split_manifest.json')['test_sample_ids_sha256'], source_artifact_sha256=pins,
                   training_protocol=FIXED500, protocol_adapter_sha256=adapter, optimizer_steps=health['optimizer_steps'],
                   stop_reason=completion['stop_reason'], study_id=STUDY['study_id'], paper_cell=False)
    module = load_evaluator500()
    record(receipt, payload)
    kw = dict(repository=Path(plan['repository']), campaign_path=Path(plan['campaign']), dataset_root=Path(plan['dataset']),
              source_dataset_root=Path(cfg['data']['root']), training_output=root, eligibility_path=receipt, eligibility_sha256=H(receipt),
              pde=pde, method=method, training_view=view)
    original = module._dataset

    class Boundary(Exception):
        pass
    calls = []

    def boundary(*args, **kwargs):
        calls.append(1); raise Boundary()
    module._dataset = boundary
    try:
        try:
            module.verify_training_artifacts(**kw)
        except Boundary:
            pass
        else:
            raise AssertionError('metadata gate did not reach the dataset boundary')
        assert calls == [1]
    finally:
        module._dataset = original
    return module, kw


def metadata(plan, row, receipt):
    verify_overlay(plan)
    if row['protocol'] == FIXED500:
        return metadata500(plan, row, receipt)
    assert row['protocol'] == SETTINGS_V3 and row['epochs'] == 200
    return same_job_eva.metadata(plan, row, receipt)


def evaluate(plan, index):
    row = plan['rows'][index]; job = os.environ.get('PDEOBS_RECEIPT_JOB') or os.environ['SLURM_JOB_ID']
    out = Path(plan['stage']) / 'receipts' / job / ('row-' + str(index)); out.mkdir(parents=True, exist_ok=True)
    try:
        provenance = allocation(plan)
        module, kw = metadata(plan, row, out / 'eligibility-candidate.json')
        record(out / 'pretest-gate.json', {'passed': True, 'test_records_touched': 0, 'identity': row['identity'], 'protocol': row['protocol'],
                                           'eligibility_sha256': kw['eligibility_sha256']})
        root = Path(plan['evaluation_gate']); root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((row['identity'] + '|' + str(Path(row['output']).resolve())).encode()).hexdigest()
        gate = root / key; gate.mkdir()
        record(gate / 'owner.json', {'identity': row['identity'], 'training_output': row['output'], 'job': job,
                                     'checkpoint_sha256': H(Path(row['output']) / 'checkpoints/last.pt'), 'stage': plan['stage'], 'claims_persist_after_failure': True})

        def verify_device():
            import torch
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1
            name = torch.cuda.get_device_name(0)
            assert any(label in name.upper() for label in plan['device_names'])
            return dict(type='cuda', name=name, visible_devices=1, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), allocation=provenance)
        module.verify_device = verify_device
        target = Path(plan['evaluation_root']) / key
        sys.argv = [str(HERE / 'mixed_eva.py')]
        for name, value in [('repository', kw['repository']), ('campaign', kw['campaign_path']), ('dataset-root', kw['dataset_root']),
                            ('source-dataset-root', kw['source_dataset_root']), ('training-output', kw['training_output']), ('evaluation-output', target),
                            ('eligibility', kw['eligibility_path']), ('eligibility-sha256', kw['eligibility_sha256']), ('pde', kw['pde']),
                            ('method', kw['method']), ('training-view', kw['training_view'])]:
            sys.argv.extend(['--' + name, str(value)])
        rc = module.main()
        record(out / 'evaluation-exit.json', {'returncode': rc, 'output': str(target), 'identity': row['identity']})
        return rc
    except BaseException as e:
        record(out / 'blocked-or-failed.json', {'identity': row['identity'], 'error': repr(e), 'traceback': traceback.format_exc(), 'automatic_retry': False})
        print('MIXED_STUDY_EVA_BLOCKED ' + row['identity'] + ' ' + repr(e), flush=True)
        return 1

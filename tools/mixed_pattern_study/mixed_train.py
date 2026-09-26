"""Mixed-pattern study trainer (2026-09-25): the pinned production one-setting-v7 runner with the study view accepted
through mixed_builder, under one of two recorded protocols chosen per plan row:

  pdeobs.fixed500-last-checkpoint/v7                        500 epochs, production Trainer, no early stopping (the original
                                                            grid protocol; the runner source is untouched except the --view
                                                            argparse choices, which mixed_builder validates instead)
  pdeobs.max200-min120-then-train-data-patience10/20260920  the settings-v3 budget200 adapter (byte-identical budget200.py):
                                                            200-epoch ceiling, train-data patience 10 after epoch 120, and the
                                                            identity's recorded training patch (scientific-patches.json)

No validation or test record is opened; the LAST durable checkpoint is the result."""
import json, pathlib, sys, types
HERE = pathlib.Path(__file__).resolve().parent
import mixed_builder  # noqa: E402  patches pdeobs.one_setting.build_experiment_config
import budget200      # noqa: E402  T2 settings-v3 adapter, byte-identical
from budget200 import H, J, PINS, canonical, replaced  # noqa: E402

FIXED500 = 'pdeobs.fixed500-last-checkpoint/v7'
SETTINGS_V3 = budget200.PROTOCOL
STUDY = J(HERE / 'study-authorization.json')
VIEW_LINE = 'parser.add_argument("--view", default="random_50pct", choices=VIEW_ORDER)'


def relax_view(source):
    return replaced(source, VIEW_LINE, 'parser.add_argument("--view", default="random_50pct")  # mixed-study: validated by mixed_builder')


def stamp(protocol, planned):
    return dict(training_protocol=protocol, planned_epochs=planned, study_id=STUDY['study_id'], study_adapter_sha256=H(__file__),
                training_view_kind=mixed_builder.KIND, mask_spec=mixed_builder.SPEC, base_view=mixed_builder.BASE_VIEW, mask_seed=STUDY.get('mask_seed'),
                seed_views=mixed_builder.SEED_VIEWS or None,
                overlay_manifest_sha256=J(HERE / 'plan.json')['overlay_manifest_sha256'], paper_cell=False)


def protocol_receipt(cfg, protocol, planned):
    return dict(protocol=protocol, adapter_sha256=H(__file__), source_pins=PINS, early_stopping=(protocol == SETTINGS_V3),
                checkpoint_selection='last', validation_used=False, test_records_touched=0, resolved_config_sha256=canonical(cfg),
                training_config=cfg['training'], mask=cfg['data']['mask'], **stamp(protocol, planned))


def load_fixed500(repository):
    repository = pathlib.Path(repository)
    for name, digest in PINS.items():
        assert H(repository / name) == digest, name
    src = repository / 'scripts/run_one_setting_v7.py'
    m = types.ModuleType('_mixed_study_frozen_runner'); m.__file__ = str(src)
    exec(compile(relax_view(src.read_text()), str(src) + '[mixed-study-fixed500]', 'exec'), m.__dict__)
    assert m.build_experiment_config is mixed_builder.build_experiment_config
    original = m.atomic_json; prepare = m._prepare

    def prepared(*args, **kwargs):
        cfg, out = prepare(*args, **kwargs)
        original(out / 'study-protocol.json', protocol_receipt(cfg, FIXED500, 500))
        return cfg, out

    def recorded(path, payload):
        if path.name in ('completion.json', 'health.json', 'history.json', 'identity.json'):
            payload = dict(payload, protocol_adapter_sha256=H(__file__), early_stopping=False, **stamp(FIXED500, 500))
            if payload.get('status') == 'completed':
                hist = payload['history'] if path.name == 'history.json' else J(path.parent / 'history.json')['history']
                assert len(hist) == 500
                payload['actual_epochs'] = len(hist); payload['stop_reason'] = 'max_epochs_500'
        return original(path, payload)
    m._prepare = prepared; m.atomic_json = recorded
    return m


def load_settings_v3(repository):
    runner_source = budget200.runner_source
    budget200.runner_source = lambda source: relax_view(runner_source(source))
    try:
        m = budget200.load_runner(repository)
    finally:
        budget200.runner_source = runner_source
    inner = m.atomic_json; prepare = m._prepare

    def prepared(*args, **kwargs):
        cfg, out = prepare(*args, **kwargs)  # budget200 writes budget-protocol.json here
        inner(out / 'study-protocol.json', protocol_receipt(cfg, SETTINGS_V3, 200))
        return cfg, out

    def recorded(path, payload):
        if path.name in ('completion.json', 'health.json', 'history.json', 'identity.json'):
            payload = dict(payload, **stamp(SETTINGS_V3, 200))  # protocol_adapter_sha256 stays budget200.py (checked by budget_eva)
        return inner(path, payload)
    m._prepare = prepared; m.atomic_json = recorded
    return m


def registry_of(p):
    import yaml
    return yaml.safe_load((pathlib.Path(p['repository']) / 'configs/method/all_pde_source_faithful_registry_v7.yaml').read_text())


def assignment(cfg, campaign, pde):
    """Component assignment of the 1800 training records: metadata only, no field data, no test record."""
    from pdeobs.runner import _dataset
    from pdeobs.one_setting import stable_split
    from pdeobs.mask_specs import parse_mask_spec
    data = _dataset(cfg, None); assert data is not None and len(data) == 2000
    problem = campaign['problem_settings'][pde]
    train, test, receipt = stable_split(data.metadata, int(campaign['seed']), f"{pde}|{problem['boundary']}|{problem['setting']}")
    if mixed_builder.KIND != 'mixture':
        return dict(kind=mixed_builder.KIND, training_records=len(train), test_records=len(test), test_sample_ids_sha256=receipt['test_sample_ids_sha256'],
                    metadata_records_read=2000, field_records_touched=0, test_records_touched=0)
    spec = parse_mask_spec(cfg['data']['mask']['protocol'])
    counts = {}; without_key = 0
    for k, pos in enumerate(train):
        md = data.metadata[pos]
        if 'regime_sample_index' not in md:
            without_key += 1
        try:
            position = int(md.get('regime_sample_index', k))
        except (TypeError, ValueError):
            position = int(k)
        stratum = (md.get('pde', ''), md.get('boundary', ''), md.get('setting', ''), md.get('regime', ''))
        index, _seed, _cycle = spec.select(mask_seed=int(cfg['seed']), stratum=stratum, position=position)
        regime = counts.setdefault(str(md.get('regime')), {})
        regime[spec.components[index].base_id] = regime.get(spec.components[index].base_id, 0) + 1
    total = {}
    for regime in counts.values():
        for cid, n in regime.items():
            total[cid] = total.get(cid, 0) + n
    return dict(per_regime=counts, total=total, training_records=len(train), test_records=len(test),
                test_sample_ids_sha256=receipt['test_sample_ids_sha256'], records_without_regime_sample_index=without_key,
                metadata_records_read=2000, field_records_touched=0, test_records_touched=0)


def audit(p, r, m):
    import torch
    pde, method, view = r['identity'].split('/')
    campaign = m.load_campaign(pathlib.Path(p['campaign'])); registry = registry_of(p)
    cfg = m.build_experiment_config(campaign, registry, dataset_root=pathlib.Path(p['dataset']), pde=pde, method=method, view=view, epochs=r['epochs'])
    assert not torch.cuda.is_initialized()
    assert cfg['training']['epochs'] == r['epochs']
    if mixed_builder.KIND == 'mixture':
        assert cfg['data']['mask'] == {'protocol': mixed_builder.SPEC}
    else:
        sv = mixed_builder.SEED_VIEWS[view]; assert cfg['training']['seed'] == int(sv['training_seed']) and cfg['data']['mask'] == dict(campaign['views'][sv['base_view']]['mask'])
    base_view = mixed_builder.BASE_VIEW if mixed_builder.KIND == 'mixture' else mixed_builder.SEED_VIEWS[view]['base_view']
    base = mixed_builder._ORIGINAL(campaign, registry, dataset_root=pathlib.Path(p['dataset']), pde=pde, method=method, view=base_view, epochs=500)
    diff = {}

    def walk(a, b, prefix):
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if isinstance(x, dict) and isinstance(y, dict) and prefix + key != 'data.mask':
                walk(x, y, prefix + key + '.')
            elif x != y:
                diff[prefix + key] = dict(frozen=y, study=x)
    walk(cfg, base, '')
    patch = r.get('training_patch') or {}
    allowed = ({'name', 'data.mask', 'training.epochs'} if mixed_builder.KIND == 'mixture' else {'name', 'training.seed'}) | {'training.' + k for k in patch}
    assert set(diff) <= allowed, sorted(set(diff) - allowed)
    for key, pair in patch.items():
        assert diff['training.' + key] == dict(frozen=pair['from'], study=pair['to']), key
    return dict(identity=r['identity'], protocol=r['protocol'], config=cfg, config_sha256=canonical(cfg), diff_vs_frozen_base_view=diff,
                mask_assignment=assignment(cfg, campaign, pde), training_records_touched=0, test_records_touched=0)


def main():
    p = J(HERE / 'plan.json'); r = p['rows'][int(sys.argv[2])]
    pde, method, view = r['identity'].split('/')
    assert mixed_builder.accepts(view) and r['mode'] == 'new'
    if r['protocol'] == FIXED500:
        assert r['epochs'] == 500 and not r.get('training_patch'); m = load_fixed500(p['repository'])
    elif r['protocol'] == SETTINGS_V3:
        assert r['epochs'] == 200; m = load_settings_v3(p['repository'])
    else:
        raise ValueError(r['protocol'])
    if sys.argv[1] == 'audit':
        print(json.dumps(audit(p, r, m))); return 0
    assert sys.argv[1] == 'run'
    sys.argv = [str(HERE / 'mixed_train.py'), '--repository', p['repository'], '--campaign', p['campaign'], '--dataset-root', p['dataset'],
                '--output', r['output'], '--pde', pde, '--method', method, '--view', view, '--epochs', str(r['epochs']), '--skip-all-view-contracts']
    return m.main()


if __name__ == '__main__':
    raise SystemExit(main())

"""Study views for the frozen one-setting-v7 config builder (mixed-pattern study and controlled seed subset, 2026-09-25).

Importing this module patches ``pdeobs.one_setting.build_experiment_config`` in-process so that two kinds of study view
are accepted, both derived from the frozen builder's output for a real paper view:
  * the mixture view (``study-authorization.json: kind = mixture``): only ``name`` and ``data.mask`` change, the latter to
    the nine-view mixture spec;
  * a seed view ``<paper view>@seed<k>`` (``kind = seed_subset``): only ``name`` and ``training.seed`` change, so the identity
    split and every sensor placement stay exactly those of the paper row and only initialisation and batch order move.
Every other key (split seed, method, optimizer, schedule, gates, horizons) is untouched. The runner script, the budget200
adapter and the frozen evaluator import the builder from ``pdeobs.one_setting`` when they are loaded, so they see the
patched function as long as this module is imported first."""
import json, pathlib
import pdeobs.dataset as _dataset_module
import pdeobs.one_setting as _one_setting
from pdeobs.mask_specs import parse_mask_spec

HERE = pathlib.Path(__file__).resolve().parent
STUDY = json.loads((HERE / 'study-authorization.json').read_text())
KIND = STUDY.get('kind', 'mixture')
MIXED_VIEW = STUDY.get('training_view')
BASE_VIEW = STUDY.get('base_view', 'random_50pct')
SPEC = STUDY.get('mask_spec')
SEED_VIEWS = {k: dict(v) for k, v in (STUDY.get('seed_views') or {}).items()}

# The overlay package (release v0.2.1 dataset.py with the additive spec branch) must be the imported one.
assert hasattr(_dataset_module.BenchmarkDataset, '_spec_mask'), 'overlay pdeobs package is not first on PYTHONPATH'
assert BASE_VIEW in _one_setting.VIEW_ORDER
if KIND == 'mixture':
    _PARSED = parse_mask_spec(SPEC)
    assert _PARSED.canonical_text() == SPEC and _PARSED.is_mixture and len(_PARSED.components) == 9
    assert all(c.weight == 1 for c in _PARSED.components)
    assert MIXED_VIEW and MIXED_VIEW not in _one_setting.VIEW_ORDER and not SEED_VIEWS
else:
    assert KIND == 'seed_subset' and SEED_VIEWS and SPEC is None and MIXED_VIEW is None
    _PARSED = None
    for name, sv in SEED_VIEWS.items():
        assert name == f"{sv['base_view']}@seed{int(sv['training_seed'])}" and sv['base_view'] in _one_setting.VIEW_ORDER
        assert int(sv['training_seed']) != int(STUDY['campaign_seed']), 'a seed view must move the trainer seed away from the campaign seed'


def accepts(view):
    return (KIND == 'mixture' and view == MIXED_VIEW) or (KIND == 'seed_subset' and view in SEED_VIEWS)


def components_match_views(campaign):
    """Each of the nine frozen views is exactly one mixture component: same protocol, same arguments."""
    wanted = {name: dict(campaign['views'][name]['mask']) for name in _one_setting.VIEW_ORDER}
    have = [dict(protocol=c.protocol, **dict(c.kwargs)) for c in _PARSED.components]
    for name, mask in wanted.items():
        assert have.count(mask) == 1, (name, mask)
    assert len(have) == len(wanted)
    return True


_ORIGINAL = getattr(_one_setting, '_mixed_study_original_builder', None) or _one_setting.build_experiment_config


def build_experiment_config(campaign, registry, *, dataset_root, pde, method, view, epochs):
    if not accepts(view):
        return _ORIGINAL(campaign, registry, dataset_root=dataset_root, pde=pde, method=method, view=view, epochs=epochs)
    if KIND == 'mixture':
        components_match_views(campaign)
        config = _ORIGINAL(campaign, registry, dataset_root=dataset_root, pde=pde, method=method, view=BASE_VIEW, epochs=epochs)
        assert config['name'] == f'v7-{pde}-{method}-{BASE_VIEW}-{int(epochs)}epoch'
        assert config['data']['mask'] == dict(campaign['views'][BASE_VIEW]['mask'])
        config['name'] = f'v7-{pde}-{method}-{MIXED_VIEW}-{int(epochs)}epoch'
        config['data']['mask'] = {'protocol': SPEC}
        return config
    sv = SEED_VIEWS[view]
    config = _ORIGINAL(campaign, registry, dataset_root=dataset_root, pde=pde, method=method, view=sv['base_view'], epochs=epochs)
    assert config['name'] == f"v7-{pde}-{method}-{sv['base_view']}-{int(epochs)}epoch" and 'seed' not in config['training']
    assert int(config['seed']) == int(STUDY['campaign_seed'])
    config['name'] = f'v7-{pde}-{method}-{view}-{int(epochs)}epoch'
    config['training']['seed'] = int(sv['training_seed'])  # TrainingConfig.seed; data split and sensors keep config["seed"]
    return config


_one_setting._mixed_study_original_builder = _ORIGINAL
_one_setting.build_experiment_config = build_experiment_config

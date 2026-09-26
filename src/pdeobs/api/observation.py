"""Unified observation specification.

Four namespaces:

``general``  -- free user parameters over the existing mask factories
                (random/grid/block/line/horizontal/vertical/boundary/clustered/
                full).  ``custom`` is listed in ``OBSERVATION_PROTOCOLS`` but is
                not resolvable here; it needs ``namespace='custom'``.  Parameters
                are validated against the real factory signatures and conflicting
                parameters are rejected.
``paper`` -- the nine frozen paper views, passed verbatim to the original
                mask construction (same protocol names, kwargs and seed policy),
                so the produced arrays are the original ones.
``custom``   -- a factory registered in ``MASK_REGISTRY`` (or an explicit mask
                array handed to ``apply_mask``), passed through unvalidated.
``stored``   -- the mask already carried by an inference input package; it
                carries ``INFERENCE_INPUT_VERSION`` rather than the general one.

An :class:`ObservationSpec` resolves to the exact ``mask`` mapping accepted by
:class:`pdeobs.dataset.BenchmarkDataset`; nothing here builds masks itself.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from . import specs as S
from ..mask_specs import is_mask_spec, parse_mask_spec


@dataclass(frozen=True)
class ObservationSpec:
    namespace: str  # general | paper | stored
    name: str  # canonical general protocol, or paper view name, or "stored"
    mask_config: Mapping[str, Any]  # exactly what BenchmarkDataset receives
    params: Mapping[str, Any] = field(default_factory=dict)
    version: str = S.OBSERVATION_GENERAL_VERSION
    label: str | None = None

    @property
    def observation_id(self) -> str:
        payload = json.dumps({"namespace": self.namespace, "name": self.name, "mask": dict(self.mask_config),
                              "version": self.version}, sort_keys=True)
        return f"{self.namespace}:{self.name}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "name": self.name, "label": self.label, "version": self.version,
                "params": dict(self.params), "mask_config": dict(self.mask_config), "observation_id": self.observation_id}


def _reject_conflicts(spec: S.ObservationProtocolSpec, params: Mapping[str, Any]) -> None:
    for group in spec.exclusive:
        given = [name for name in group if params.get(name) is not None]
        if len(given) > 1:
            raise ValueError(f"observation {spec.name!r}: parameters {given} conflict; give only one of {list(group)}")


def make_observation(protocol: str | Mapping[str, Any] | ObservationSpec | None = None, /, *,
                     namespace: str | None = None, **params: Any) -> ObservationSpec:
    """Build an observation specification.

    Examples::

        make_observation("random", ratio=0.5)
        make_observation("boundary", width=3)
        make_observation("random_65pct", namespace="paper")   # or make_observation("paper:R65")
        make_observation({"protocol": "demo_checkerboard"}, namespace="custom")
        make_observation("stored")   # use the mask carried by an inference input package
    """
    if isinstance(protocol, ObservationSpec):
        if params or (namespace and namespace != protocol.namespace):
            raise ValueError("cannot modify an already resolved ObservationSpec; build a new one")
        return protocol
    if isinstance(protocol, Mapping):
        mapping = dict(protocol)
        if params:
            raise ValueError("pass parameters either inside the mapping or as keywords, not both")
        ns = namespace or mapping.pop("namespace", None) or "general"
        name = mapping.pop("protocol", None) or mapping.pop("name", None) or mapping.pop("view", None)
        if name is None:
            raise ValueError("observation mapping needs a 'protocol' (general/custom) or 'view' (paper) key")
        return make_observation(str(name), namespace=ns, **mapping)
    if protocol is None:
        raise ValueError("an observation protocol name is required")
    text = str(protocol).strip()
    if ":" in text and namespace is None:
        namespace, text = text.split(":", 1)
    namespace = (namespace or "general").strip().lower().replace("_", "-")
    if namespace in {"paper"}:
        return _paper_view(text, params)
    if namespace == "stored":
        if params:
            raise ValueError("the stored observation takes no parameters; it uses the mask inside the input package")
        return ObservationSpec("stored", "stored", {"protocol": "stored"}, {}, S.INFERENCE_INPUT_VERSION, "stored")
    if namespace == "custom":
        seed = params.pop("seed", None)
        clean = {k: v for k, v in params.items() if v is not None}
        config = {"protocol": text, **clean}
        return ObservationSpec("custom", text, config, {**clean, **({"seed": seed} if seed is not None else {})},
                               S.OBSERVATION_GENERAL_VERSION, text)
    if namespace != "general":
        raise ValueError(f"unknown observation namespace {namespace!r}; use general, paper, custom or stored")
    if text.lower() == "stored":
        return make_observation("stored", namespace="stored")
    if text in S.PAPER_VIEWS or text in S.PAPER_VIEW_LABELS:
        raise ValueError(f"{text!r} is a paper view; request it with namespace='paper' (or 'paper:{text}') "
                         "so the frozen definition is used explicitly")
    if is_mask_spec(text):
        # v0.2.1: a protocol with an observed percentage, or a "+"-joined mixture; the spec carries
        # its own parameters and is handed to BenchmarkDataset verbatim (see docs/observation_specs.md).
        if params:
            raise ValueError("a mask spec carries its own parameters; write them inside the spec string "
                             "(e.g. 'line_sensors(num_lines=64, orientation=horizontal) + uniform 20'), not as extra parameters")
        parsed = parse_mask_spec(text)
        canonical = parsed.canonical_text()
        return ObservationSpec("general", parsed.view_id, {"protocol": canonical}, {"spec": canonical},
                               S.OBSERVATION_GENERAL_VERSION, canonical)
    name = S.normalize_observation_name(text)
    spec = S.OBSERVATION_PROTOCOLS[name]
    if name == "custom":
        raise ValueError("use namespace='custom' with the registered factory name for custom masks")
    known = {p.name for p in spec.params}
    unknown = sorted(set(params) - known - {"seed"})
    if unknown:
        raise ValueError(f"observation {name!r} does not accept {unknown}; valid parameters: {sorted(known)} (+ seed)")
    validated: dict[str, Any] = {}
    for p in spec.params:
        if p.name in params:
            validated[p.name] = p.validate(params[p.name], f"observation {name}")
    _reject_conflicts(spec, validated)
    if name == "full" and validated:
        raise ValueError("the full observation accepts no parameters")
    registry_name, fixed = S._GENERAL_TO_REGISTRY[name]
    # accept the finer random registry names (random_1pct, ...) verbatim when given
    raw = str(protocol).strip().lower()
    if raw in {"random_1pct", "random_5pct", "random_10pct"}:
        registry_name = raw
    config: dict[str, Any] = {"protocol": registry_name, **fixed}
    config.update({k: v for k, v in validated.items() if v is not None})
    if name == "block" and config.get("block_shape") is not None:
        # the public parameter is one side length; the mask factory takes (height, width)
        side = int(config["block_shape"])
        config["block_shape"] = (side, side)
    label = name
    return ObservationSpec("general", name, config, validated, S.OBSERVATION_GENERAL_VERSION, label)


def _paper_view(text: str, params: Mapping[str, Any]) -> ObservationSpec:
    if params:
        raise ValueError("paper views are frozen; they accept no parameters (use the general namespace to vary them)")
    key = text.strip()
    if key in S.PAPER_VIEW_LABELS:
        key = S.PAPER_VIEW_LABELS[key]
    key = key.lower()
    if key not in S.PAPER_VIEWS:
        raise ValueError(f"unknown paper view {text!r}; views: {', '.join(S.PAPER_VIEWS)} "
                         f"(labels {', '.join(S.PAPER_VIEW_LABELS)})")
    view = S.PAPER_VIEWS[key]
    return ObservationSpec("paper", key, dict(view["mask"]), {}, S.OBSERVATION_PAPER_VERSION, view["label"])


def paper_views() -> list[ObservationSpec]:
    return [_paper_view(name, {}) for name in S.PAPER_VIEWS]


def realized_count(spec: ObservationSpec, shape: tuple[int, int], *, seed: int = 0) -> int:
    """Number of observed cells the resolved configuration actually produces on ``shape``."""
    from ..masks import generate_mask

    config = dict(spec.mask_config)
    protocol = config.pop("protocol")
    if protocol == "full":
        return int(np.prod(shape))
    if protocol == "stored":
        raise ValueError("stored observations have no synthetic count; inspect the input package mask")
    return int(generate_mask(protocol, shape, seed=seed, **config).sum())


def describe_observations() -> dict[str, Any]:
    general = {name: {"doc": s.doc, "aliases": list(s.aliases),
                      "params": [{"name": p.name, "type": p.type, "default": p.default, "doc": p.doc} for p in s.params],
                      "exclusive": [list(g) for g in s.exclusive]} for name, s in S.OBSERVATION_PROTOCOLS.items()}
    paper = {name: {"label": v["label"], "mask": v["mask"], "expected_count_128": v["expected_count_128"]}
             for name, v in S.PAPER_VIEWS.items()}
    return {"general": general, "paper": paper, "general_version": S.OBSERVATION_GENERAL_VERSION,
            "paper_version": S.OBSERVATION_PAPER_VERSION,
            "notes": ["counts are exact discrete cell counts; ratios are targets, not guarantees",
                      "grid/line/boundary geometry is discrete: the realized fraction may differ from the requested ratio",
                      "paper views are frozen 128x128 definitions; small demo grids must not reuse the fixed 128 counts",
                      "block observes everything outside a non-wrapping missing square; boundary is an outer array band, not a boundary condition",
                      "mask (observation protocol) and geometry (solid/obstacle field) are different inputs and are never interchanged"]}

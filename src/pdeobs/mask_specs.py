"""Mask specifications: protocol name plus parameter shorthand, and deterministic mixtures.

A *mask spec* is a short string accepted wherever a mask protocol name is accepted, both for
training (``data.mask.protocol``) and for test views (``evaluation.observation_protocols``)::

    uniform 1                       one percent of the cells observed at random
    line 30 + random 20             a mixture: line sensors covering 30 percent, random 20 percent
    uniform 1 x2 + sensor 20 + block 50
                                    weighted mixture (uniform twice as often as the others)
    line_sensors(num_lines=64, orientation=horizontal)
                                    a registered protocol with explicit keyword arguments

Grammar.  Components are separated by ``+``.  A component is a name, then optionally either an
*observed percentage* (``uniform 1``, ``block 50``, ``line 30``) or an explicit keyword list in
parentheses, then optionally an integer weight written ``x2`` or ``*2``.  Names are the registered
protocols and their registry aliases plus the shorthand families below.  The number after a
shorthand is always the fraction of cells that end up *observed*, in percent; it is converted to the
protocol's own parameter at generation time using the field's spatial shape.

Shorthand families (the resolved protocol is what seeds the mask, exactly as for a fixed view):

    uniform, random, rand, u        -> random_3pct        ratio = p / 100
    sensor(s), cluster(s), clustered -> clustered_sensors ratio = p / 100
    block(s), missing               -> block_missing      missing_fraction = 1 - p / 100
    grid, regular                   -> regular_grid       ratio = p / 100 (spacing = round(1 / sqrt(ratio)))
    boundary, edge, band            -> boundary_sensors   width solved so the band covers p percent
    line(s)                         -> line_sensors       num_lines solved for p percent, both orientations
    hline(s), horizontal            -> line_sensors       horizontal lines, num_lines = round(p / 100 * H)
    vline(s), vertical              -> line_sensors       vertical lines,   num_lines = round(p / 100 * W)
    full                            -> every cell observed

A mixture assigns one component per sample, deterministically: the weight-expanded component cycle
is rotated by a seed derived from the dataset mask seed, the canonical spec and the sample's factor
stratum, and indexed by the sample's stratum position.  The per-sample mask seed is then derived
from the *resolved protocol name* alone, so a component such as ``random 20`` produces the same mask
for a given sample whether it is requested on its own or inside a mixture.

Nothing here changes a registered protocol, a registered alias, the nine paper views or any seed
derivation of an existing configuration: a plain registered name still takes the legacy path.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .masks import MASK_PROTOCOL_NAMES, generate_mask
from .registry import MASK_REGISTRY, normalize_name
from .schema import derive_seed

__all__ = [
    "MaskComponent",
    "MaskSpec",
    "SHORTHAND_FAMILIES",
    "canonical_view_id",
    "is_mask_spec",
    "parse_mask_spec",
    "realized_observed_fraction",
]

# family word -> (registered protocol, implied keyword arguments)
SHORTHAND_FAMILIES: dict[str, tuple[str, dict[str, Any]]] = {
    "uniform": ("random_3pct", {}),
    "random": ("random_3pct", {}),
    "rand": ("random_3pct", {}),
    "u": ("random_3pct", {}),
    "sensor": ("clustered_sensors", {}),
    "sensors": ("clustered_sensors", {}),
    "cluster": ("clustered_sensors", {}),
    "clusters": ("clustered_sensors", {}),
    "clustered": ("clustered_sensors", {}),
    "block": ("block_missing", {}),
    "blocks": ("block_missing", {}),
    "missing": ("block_missing", {}),
    "grid": ("regular_grid", {}),
    "regular": ("regular_grid", {}),
    "boundary": ("boundary_sensors", {}),
    "edge": ("boundary_sensors", {}),
    "band": ("boundary_sensors", {}),
    "line": ("line_sensors", {"orientation": "both"}),
    "lines": ("line_sensors", {"orientation": "both"}),
    "hline": ("line_sensors", {"orientation": "horizontal"}),
    "hlines": ("line_sensors", {"orientation": "horizontal"}),
    "horizontal": ("line_sensors", {"orientation": "horizontal"}),
    "vline": ("line_sensors", {"orientation": "vertical"}),
    "vlines": ("line_sensors", {"orientation": "vertical"}),
    "vertical": ("line_sensors", {"orientation": "vertical"}),
    "full": ("full", {}),
}

# which parameter an observed percentage becomes, per registered protocol
_PERCENT_PARAMETER: dict[str, str] = {
    "random_1pct": "ratio",
    "random_3pct": "ratio",
    "random_5pct": "ratio",
    "random_10pct": "ratio",
    "regular_grid": "ratio",
    "clustered_sensors": "ratio",
    "block_missing": "missing_fraction",
    "boundary_sensors": "width",
    "line_sensors": "num_lines",
}

_COMPONENT = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_\-]*)"
    r"\s*(?:\((?P<kwargs>[^()]*)\))?"
    r"\s*(?P<percent>\d+(?:\.\d+)?)?\s*%?"
    r"\s*(?:[xX*]\s*(?P<weight>\d+))?\s*$"
)
_TRAILING_NUMBER = re.compile(r"^(?P<name>[A-Za-z][A-Za-z_\-]*?)(?P<percent>\d+(?:\.\d+)?)$")
_ID_CLEAN = re.compile(r"[^a-z0-9._]+")   # protocol names keep their underscores; "-" separates fields


def _registered(name: str) -> str | None:
    """Return the canonical registered protocol for ``name`` or None."""

    try:
        canonical = MASK_REGISTRY.resolve_name(name)
    except ValueError:
        return None
    return canonical if canonical in MASK_REGISTRY.names() else None


def _resolve_name(raw: str) -> tuple[str, dict[str, Any]]:
    key = normalize_name(raw)
    if key in SHORTHAND_FAMILIES:
        protocol, implied = SHORTHAND_FAMILIES[key]
        return protocol, dict(implied)
    registered = _registered(raw)
    if registered is not None:
        return registered, {}
    raise ValueError(
        f"unknown mask protocol or shorthand {raw!r}; registered protocols: "
        f"{', '.join(MASK_PROTOCOL_NAMES)}; shorthand families: {', '.join(sorted(SHORTHAND_FAMILIES))}"
    )


def _parse_kwargs(text: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"keyword argument {item!r} must be written key=value")
        key, value = (part.strip() for part in item.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid keyword name {key!r}")
        try:
            parsed: Any = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        if isinstance(parsed, (list, tuple)):
            parsed = tuple(parsed)
        kwargs[key] = parsed
    return kwargs


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return "x".join(_format_value(item) for item in value)
    return str(value)


def _solve_boundary_width(shape: tuple[int, int], fraction: float) -> int:
    height, width = shape
    best, best_error = 1, float("inf")
    for candidate in range(1, max(1, min(shape) // 2) + 1):
        realized = 1.0 - ((height - 2 * candidate) * (width - 2 * candidate)) / (height * width)
        error = abs(realized - fraction)
        if error < best_error:
            best, best_error = candidate, error
    return best


def _solve_line_count(shape: tuple[int, int], fraction: float, orientation: str) -> int:
    height, width = shape
    if orientation == "horizontal":
        return max(1, min(height, int(round(fraction * height))))
    if orientation == "vertical":
        return max(1, min(width, int(round(fraction * width))))
    best, best_error = 1, float("inf")
    for candidate in range(1, height + width + 1):
        rows, columns = (candidate + 1) // 2, candidate // 2
        if rows > height or columns > width:
            break
        realized = 1.0 - (1.0 - rows / height) * (1.0 - columns / width)
        error = abs(realized - fraction)
        if error < best_error:
            best, best_error = candidate, error
    return best


@dataclass(frozen=True)
class MaskComponent:
    """One resolved component: a registered protocol, its arguments and a weight."""

    protocol: str
    kwargs: tuple[tuple[str, Any], ...] = ()
    percent: float | None = None
    weight: int = 1
    family: str | None = field(default=None, compare=False)   # presentation only

    @property
    def component_id(self) -> str:
        parts = [self.protocol]
        for key, value in self.kwargs:
            parts.append(f"{key}-{_format_value(value)}")
        if self.percent is not None:
            parts.append(f"p{self.percent:g}")
        text = "-".join(parts)
        text = _ID_CLEAN.sub("-", text.lower()).strip("-")
        return text + (f"-x{self.weight}" if self.weight != 1 else "")

    @property
    def base_id(self) -> str:
        """The component identity without its weight."""

        return MaskComponent(self.protocol, self.kwargs, self.percent, 1, self.family).component_id

    def canonical_text(self) -> str:
        head = self.protocol
        if self.kwargs:
            head += "(" + ", ".join(f"{key}={_format_value(value)}" for key, value in self.kwargs) + ")"
        if self.percent is not None:
            head += f" {self.percent:g}"
        if self.weight != 1:
            head += f" x{self.weight}"
        return head

    def target_fraction(self) -> float | None:
        """The observed fraction this component aims for, when it is declared."""

        if self.protocol == "full":
            return 1.0
        if self.percent is not None:
            return self.percent / 100.0
        kwargs = dict(self.kwargs)
        if self.protocol == "block_missing" and "missing_fraction" in kwargs:
            return 1.0 - float(kwargs["missing_fraction"])
        if "ratio" in kwargs and self.protocol in _PERCENT_PARAMETER:
            return float(kwargs["ratio"])
        if self.protocol in {"random_1pct", "random_3pct", "random_5pct", "random_10pct"} and not kwargs:
            return {"random_1pct": 0.01, "random_3pct": 0.03, "random_5pct": 0.05, "random_10pct": 0.10}[self.protocol]
        return None

    def resolved_kwargs(self, shape: tuple[int, int]) -> dict[str, Any]:
        """Keyword arguments for ``generate_mask`` at the given spatial shape."""

        kwargs = dict(self.kwargs)
        if self.percent is None or self.protocol == "full":
            return kwargs
        fraction = self.percent / 100.0
        parameter = _PERCENT_PARAMETER[self.protocol]
        if parameter == "ratio":
            kwargs["ratio"] = fraction
        elif parameter == "missing_fraction":
            kwargs["missing_fraction"] = 1.0 - fraction
        elif parameter == "width":
            kwargs["width"] = _solve_boundary_width(shape, fraction)
        elif parameter == "num_lines":
            orientation = str(kwargs.get("orientation", "both"))
            kwargs["num_lines"] = _solve_line_count(shape, fraction, orientation)
        return kwargs

    def generate(self, shape: tuple[int, int], seed: int) -> Any:
        if self.protocol == "full":
            import numpy as np

            return np.ones(shape, dtype=bool)
        return generate_mask(self.protocol, shape, seed=seed, **self.resolved_kwargs(shape))

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.component_id,
            "protocol": self.protocol,
            "kwargs": dict(self.kwargs),
            "percent": self.percent,
            "weight": self.weight,
            "family": self.family,
            "target_observed_fraction": self.target_fraction(),
        }


@dataclass(frozen=True)
class MaskSpec:
    """A parsed spec: one or more components with a deterministic selection cycle."""

    components: tuple[MaskComponent, ...]
    source: str = field(default="", compare=False)

    @property
    def is_mixture(self) -> bool:
        return len(self.cycle) > 1

    @property
    def cycle(self) -> tuple[int, ...]:
        order: list[int] = []
        for index, component in enumerate(self.components):
            order.extend([index] * component.weight)
        return tuple(order)

    @property
    def view_id(self) -> str:
        return "__".join(component.component_id for component in self.components)

    def canonical_text(self) -> str:
        return " + ".join(component.canonical_text() for component in self.components)

    def target_fraction(self) -> float | None:
        """Weight-averaged observed fraction when every component declares one."""

        fractions = [component.target_fraction() for component in self.components]
        if any(value is None for value in fractions):
            return None
        total = sum(component.weight for component in self.components)
        return sum(f * c.weight for f, c in zip(fractions, self.components, strict=True)) / total

    def select(self, *, mask_seed: int, stratum: Sequence[object], position: int) -> tuple[int, int, int]:
        """Return (component index, selection seed, cycle position) for one sample."""

        cycle = self.cycle
        offset = derive_seed(mask_seed, "mask_mixture", self.canonical_text(), *stratum)
        cycle_position = (int(position) + int(offset)) % len(cycle)
        return cycle[cycle_position], int(offset), cycle_position

    def describe(self) -> dict[str, Any]:
        return {
            "spec": self.canonical_text(),
            "view_id": self.view_id,
            "mixture": self.is_mixture,
            "components": [component.describe() for component in self.components],
            "cycle": [self.components[index].base_id for index in self.cycle],
            "target_observed_fraction": self.target_fraction(),
        }


def _split_components(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses in mask spec")
        if character == "+" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    if depth != 0:
        raise ValueError("unbalanced parentheses in mask spec")
    parts.append("".join(current))
    return parts


def _component_from_text(text: str) -> MaskComponent:
    match = _COMPONENT.match(text)
    if not match:
        raise ValueError(f"cannot parse mask component {text.strip()!r}")
    name = match.group("name")
    percent_text = match.group("percent")
    kwargs_text = match.group("kwargs")
    weight = int(match.group("weight") or 1)
    if percent_text is None and kwargs_text is None:
        trailing = _TRAILING_NUMBER.match(name)
        if trailing and _registered(name) is None and normalize_name(name) not in SHORTHAND_FAMILIES:
            name, percent_text = trailing.group("name"), trailing.group("percent")
    return _component_from_parts(name, percent_text, kwargs_text, weight)


def _component_from_parts(
    name: str, percent_text: str | float | int | None, kwargs_text: str | Mapping[str, Any] | None, weight: int
) -> MaskComponent:
    protocol, implied = _resolve_name(name)
    explicit = _parse_kwargs(kwargs_text) if isinstance(kwargs_text, str) else dict(kwargs_text or {})
    percent: float | None = None
    if percent_text is not None:
        percent = float(percent_text)
        if protocol == "full":
            raise ValueError("'full' does not take a percentage")
        parameter = _PERCENT_PARAMETER.get(protocol)
        conflicts = {parameter, "count", "spacing", "block_shape"} & set(explicit)
        if conflicts:
            raise ValueError(
                f"component {name!r}: the observed percentage already sets {sorted(conflicts)}; "
                "give either the percentage or those keyword arguments"
            )
        if not 0.0 < percent <= 100.0:
            raise ValueError(f"observed percentage must lie in (0, 100], got {percent:g}")
        if protocol == "block_missing" and percent >= 100.0:
            raise ValueError("a block component needs an observed percentage below 100")
        if protocol not in _PERCENT_PARAMETER:
            raise ValueError(f"protocol {protocol!r} does not accept an observed percentage")
    if weight < 1:
        raise ValueError("component weights must be positive integers")
    kwargs = {**implied, **explicit}
    family = normalize_name(name) if normalize_name(name) in SHORTHAND_FAMILIES else None
    return MaskComponent(
        protocol=protocol,
        kwargs=tuple(sorted(kwargs.items())),
        percent=percent,
        weight=int(weight),
        family=family,
    )


def _finish(components: Sequence[MaskComponent], source: str) -> MaskSpec:
    if not components:
        raise ValueError("a mask spec needs at least one component")
    seen: set[str] = set()
    for component in components:
        if component.base_id in seen:
            raise ValueError(
                f"duplicate component {component.canonical_text()!r}; use a weight (x2) instead of repeating it"
            )
        seen.add(component.base_id)
    return MaskSpec(components=tuple(components), source=source)


def parse_mask_spec(spec: str | Mapping[str, Any] | Sequence[Any]) -> MaskSpec:
    """Parse a spec string, a ``{"protocol": "mixture", "components": [...]}`` mapping, or a list."""

    if isinstance(spec, str):
        return _finish([_component_from_text(part) for part in _split_components(spec)], spec)
    if isinstance(spec, Mapping):
        options = dict(spec)
        protocol = str(options.pop("protocol", "mixture")).strip().lower()
        components = options.pop("components", None)
        if options:
            raise ValueError(f"unexpected mixture options {sorted(options)}; put arguments on each component")
        if protocol not in {"mixture", "mix", "mixed"} or components is None:
            raise ValueError("a mapping spec needs protocol: mixture and a components list")
        return parse_mask_spec(list(components))
    parsed: list[MaskComponent] = []
    for item in spec:
        if isinstance(item, str):
            parsed.extend(_component_from_text(part) for part in _split_components(item))
        elif isinstance(item, Mapping):
            options = dict(item)
            name = options.pop("protocol", None) or options.pop("name", None)
            if name is None:
                raise ValueError("each component mapping needs a protocol")
            percent = options.pop("percent", None)
            weight = int(options.pop("weight", 1))
            parsed.append(_component_from_parts(str(name), percent, options, weight))
        else:
            raise ValueError(f"unsupported component {item!r}")
    return _finish(parsed, " + ".join(component.canonical_text() for component in parsed))


def is_mask_spec(protocol: str) -> bool:
    """True when ``protocol`` is not a registered name and parses as a spec."""

    if not isinstance(protocol, str) or not protocol.strip():
        return False
    if _registered(protocol) is not None:
        return False
    if normalize_name(protocol) in {"full", "full_observation", "all_visible", "mixed_partial", "balanced_partial", "all_partial"}:
        return False
    try:
        parse_mask_spec(protocol)
    except ValueError:
        return False
    return True


def canonical_view_id(protocol: str) -> str:
    """Stable identifier for a test view: registered names unchanged, specs slugged."""

    if not is_mask_spec(protocol):
        return str(protocol)
    return parse_mask_spec(protocol).view_id


def realized_observed_fraction(component: MaskComponent, shape: tuple[int, int], seed: int = 0) -> float:
    """Observed fraction a component actually realizes at ``shape`` (for tests and reports)."""

    mask = component.generate(shape, seed)
    return float(mask.mean())


def describe_mask_config(mask: Mapping[str, Any]) -> dict[str, Any] | None:
    """Describe a ``data.mask`` mapping when it is a spec; None for legacy configurations."""

    options = dict(mask)
    requested = str(options.get("protocol", "random_3pct"))
    if normalize_name(requested) in {"mixture", "mix", "mixed"} and "components" in options:
        return parse_mask_spec(options).describe()
    if len(options) == 1 and is_mask_spec(requested):
        return parse_mask_spec(requested).describe()
    return None


"""Generate the v0.2.0 reference docs from the live specs (kept in sync with the code)."""
import sys
sys.path.insert(0, "src")
from pdeobs.api import specs as S  # noqa: E402
from pdeobs.api import observation as O  # noqa: E402

DOCS = "docs"


def model_reference() -> None:
    L = ["# Model parameter reference (v0.2.0)", "",
         "Every parameter is validated by the easy API before any expensive work: unknown keys, wrong types,",
         "out-of-range values, alias confusion (`width` vs `hidden`, `depth` vs `layers`, `physics_slices` vs",
         "`slices`) and illegal combinations are rejected with the list of valid options. A parameter that",
         "cannot be mapped to the real constructor is never silently ignored.", "",
         "`geometry_channels` defaults to 1 (the paper setting, 1=solid); set 0 for models that accept it to",
         "run without a geometry input. `in_channels`/`out_channels` are taken from the dataset unless overridden.", ""]
    for name, s in S.MODEL_SPECS.items():
        if s.upstream_wrapper:
            continue
        L += [f"## {name} — {s.label}", "",
              f"- family: {s.family}; tasks: {', '.join(s.tasks)}; rollout adapter: {s.temporal_wrapper or 'native'}; requires torch: {s.requires_torch}"]
        if s.aliases:
            L.append(f"- aliases: {', '.join(s.aliases)}")
        if s.main_paper_model:
            L.append("- **main paper model**")
        L += ["", "| parameter | type | default | range/choices | effect | meaning |", "|---|---|---|---|---|---|"]
        for p in s.params:
            rng = ""
            if p.choices:
                rng = "{" + ", ".join(map(str, p.choices)) + "}"
            elif p.minimum is not None or p.maximum is not None:
                rng = f"[{p.minimum}, {p.maximum}]"
            L.append(f"| `{p.name}` | {p.type} | {p.default} | {rng} | {p.effect} | {p.doc} |")
        L.append("")
        if s.constraints:
            L.append("Constraints:")
            L += [f"- {c}" for c in s.constraints]
            L.append("")
        if s.presets:
            L.append("Presets: " + "; ".join(f"`{k}` = {dict(v)}" for k, v in s.presets.items()))
            if s.temporal_presets:
                L.append("")
                L.append("Temporal presets (rollout): " + "; ".join(f"`{k}` = {dict(v)}" for k, v in s.temporal_presets.items()))
            L.append("")
        if s.notes:
            L += [f"Notes: {s.notes}", ""]
    open(f"{DOCS}/model_parameter_reference.md", "w", encoding="utf-8", newline="\n").write("\n".join(L))


def observation_reference() -> None:
    d = O.describe_observations()
    L = ["# Observation reference (v0.2.0)", "",
         "Four namespaces. `general` exposes free parameters over the existing mask factories (`custom` is",
         "listed below but is not resolvable from `general`); `paper` replays the nine frozen paper views",
         "verbatim (same protocol, kwargs and seed policy); `custom` resolves a factory registered in",
         "`MASK_REGISTRY`; `stored` reuses the mask already carried by an inference input package.", "",
         "General notes:"]
    L += [f"- {n}" for n in d["notes"]]
    L += ["", "## general protocols", "", "| protocol | aliases | parameters | meaning |", "|---|---|---|---|"]
    for name, s in d["general"].items():
        params = "; ".join(f"`{p['name']}`={p['default']}" for p in s["params"]) or "(none)"
        L.append(f"| `{name}` | {', '.join(s['aliases']) or '-'} | {params} | {s['doc']} |")
    L += ["", "Conflicting parameters (give at most one of each group) are rejected:", ""]
    for name, s in d["general"].items():
        if s["exclusive"]:
            L.append(f"- `{name}`: " + "; ".join("{" + ", ".join(g) + "}" for g in s["exclusive"]))
    L += ["", "## paper views (128×128, frozen)", "",
          "Request with `paper:<view>` or the label, e.g. `paper:R65`. These carry no parameters.", "",
          "| view | label | mask config | observed cells @128² |", "|---|---|---|---|"]
    for name, v in d["paper"].items():
        L.append(f"| `{name}` | {v['label']} | `{v['mask']}` | {v['expected_count_128']} |")
    L += ["", f"general schema: `{d['general_version']}` (also used by `custom`); paper schema: "
          f"`{d['paper_version']}`; stored schema: `{S.INFERENCE_INPUT_VERSION}`", ""]
    open(f"{DOCS}/observation_reference.md", "w", encoding="utf-8", newline="\n").write("\n".join(L))


model_reference()
observation_reference()
print("wrote model_parameter_reference.md and observation_reference.md")

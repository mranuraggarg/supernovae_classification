"""Export compact-feature definitions for the Phase 2 Tier 2."""

from __future__ import annotations

import ast
import csv
import json
import os
from typing import Any


TIER2_RESULTS_DIR = "results/phase2_tier2"
FEATURE_DICTIONARY_CSV = f"{TIER2_RESULTS_DIR}/compact_feature_dictionary.csv"
FEATURE_DICTIONARY_JSON = f"{TIER2_RESULTS_DIR}/compact_feature_dictionary.json"
FEATURE_DICTIONARY_MD = f"{TIER2_RESULTS_DIR}/compact_feature_dictionary.md"

TIER2_COMMON_PATH = "phase2_tier2_common.py"
FEATURE_REGISTRY_PATH = "feature_pipeline/extraction/feature_registry.py"
INTERPRETATION_TABLE_PATH = "results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.csv"


FORMULA_OVERRIDES = {
    "time_span": "max(normalized observation time) - min(normalized observation time)",
}

CAUTION_NOTES = {
    "time_span": "This is observational time coverage, not a measured rise time or decline time.",
    "i_amplitude": "This is a reconstructed i-band peak-to-trough amplitude proxy, not a direct bolometric amplitude.",
}

PHYSICAL_INTERPRETATION_BY_GROUP = {
    "brightness": "Brightness-scale proxy related to observed luminosity structure and survey-frame flux behavior.",
    "color": "Peak color proxy related to temperature, spectral slope, and band-to-band flux contrast.",
    "variability": "Light-curve shape or variability proxy summarizing flux spread or peak-to-trough contrast.",
    "temporal": "Timing or temporal-coverage proxy describing when peak flux occurs or how long the object is observed.",
}

INTERPRETATION_OVERRIDES = {
    "i_amplitude": "i-band amplitude captures reconstructed peak-to-trough flux contrast as a light-curve-shape proxy.",
}


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def read_python_assignment(path: str, name: str) -> Any:
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"Could not find assignment {name!r} in {path}.")


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def feature_to_group_map(feature_groups: dict[str, list[str]]) -> dict[str, str]:
    group_by_feature = {}
    for group, features in feature_groups.items():
        for feature in features:
            group_by_feature[feature] = group
    return group_by_feature


def registry_definition_map() -> dict[str, dict[str, str]]:
    namespace: dict[str, Any] = {}
    with open(FEATURE_REGISTRY_PATH) as handle:
        exec(compile(handle.read(), FEATURE_REGISTRY_PATH, "exec"), namespace)
    return {spec.name: spec.to_dict() for spec in namespace["FEATURE_REGISTRY"]}


def formula_from_definition(feature: str, definition: str) -> str:
    if feature in FORMULA_OVERRIDES:
        return FORMULA_OVERRIDES[feature]
    if feature.startswith("peak_color_"):
        bands = feature.removeprefix("peak_color_").split("_minus_")
        if len(bands) == 2:
            return f"-2.5 * log10({bands[0]}_peak_flux / {bands[1]}_peak_flux), clipped to [-5, 5]"
    if feature.endswith("_peak_flux"):
        band = feature.split("_", 1)[0]
        return f"log10(1 + max(reconstructed {band}-band flux, 0))"
    if feature.endswith("_mean_flux"):
        band = feature.split("_", 1)[0]
        return f"sign(mean {band}-band flux) * log10(1 + abs(mean {band}-band flux))"
    if feature.endswith("_std_flux"):
        band = feature.split("_", 1)[0]
        return f"log10(1 + standard deviation of reconstructed {band}-band flux)"
    if feature.endswith("_amplitude"):
        band = feature.split("_", 1)[0]
        return f"log10(1 + max peak-to-trough reconstructed {band}-band amplitude)"
    if feature.endswith("_time_of_peak"):
        band = feature.split("_", 1)[0]
        return f"normalized time coordinate at maximum reconstructed {band}-band flux"
    return definition


def build_dictionary_rows() -> list[dict[str, Any]]:
    compact_features = read_python_assignment(TIER2_COMMON_PATH, "COMPACT_FEATURES")
    feature_groups = read_python_assignment(TIER2_COMMON_PATH, "FEATURE_GROUPS")
    group_by_feature = feature_to_group_map(feature_groups)
    registry_by_feature = registry_definition_map()
    interpretation_by_feature = {
        row["feature"]: row
        for row in read_csv_rows(INTERPRETATION_TABLE_PATH)
        if row.get("feature")
    }

    rows = []
    for order, feature in enumerate(compact_features, start=1):
        registry = registry_by_feature.get(feature, {})
        interpretation = interpretation_by_feature.get(feature, {})
        group = group_by_feature.get(feature, registry.get("group", "unknown"))
        definition = registry.get("definition", "")
        formula = formula_from_definition(feature, definition)
        rows.append(
            {
                "order": order,
                "feature": feature,
                "feature_group": group,
                "definition": definition,
                "formula_or_computation": formula,
                "physical_interpretation": INTERPRETATION_OVERRIDES.get(
                    feature,
                    interpretation.get(
                        "interpretation",
                        PHYSICAL_INTERPRETATION_BY_GROUP.get(group, ""),
                    ),
                ),
                "directional_interpretation": interpretation.get("directional_interpretation", ""),
                "importance_label": interpretation.get("relative_importance", ""),
                "shap_rank": interpretation.get("shap_rank", ""),
                "manuscript_caution": CAUTION_NOTES.get(feature, ""),
            }
        )
    return rows


def write_markdown(path: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Compact Feature Dictionary",
        "",
        "This table defines the 16 engineered features used by the compact Phase 2 Tier 2 model.",
        "",
        "## Supporting cautions",
        "",
        "- `time_span` is observational time coverage, not a rise-time or decline-time measurement.",
        "- Flux-scale, spread, and amplitude features are engineered log-compressed summaries of reconstructed light curves.",
        "- Color features are magnitude-style peak-flux-ratio proxies and should be described as color proxies rather than direct spectroscopic colors.",
        "",
        "## Feature table",
    ]
    lines.extend(
        markdown_table(
            [
                "order",
                "feature",
                "group",
                "definition",
                "formula / computation",
                "interpretation",
                "caution",
            ],
            [
                [
                    str(row["order"]),
                    row["feature"],
                    row["feature_group"],
                    row["definition"],
                    row["formula_or_computation"],
                    row["physical_interpretation"],
                    row["manuscript_caution"],
                ]
                for row in rows
            ],
        )
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    rows = build_dictionary_rows()
    payload = {
        "artifact": "phase2_tier2_compact_feature_dictionary",
        "source_files": {
            "tier2_common": TIER2_COMMON_PATH,
            "feature_registry": FEATURE_REGISTRY_PATH,
            "interpretation_table": INTERPRETATION_TABLE_PATH,
        },
        "feature_count": len(rows),
        "rows": rows,
        "notes": [
            "time_span is observational time coverage, not a rise-time or decline-time feature.",
            "Color features are peak-flux-ratio proxies clipped to [-5, 5].",
            "Definitions are derived from the existing feature registry and compact interpretation table.",
        ],
    }
    fieldnames = [
        "order",
        "feature",
        "feature_group",
        "definition",
        "formula_or_computation",
        "physical_interpretation",
        "directional_interpretation",
        "importance_label",
        "shap_rank",
        "manuscript_caution",
    ]
    write_csv(FEATURE_DICTIONARY_CSV, fieldnames, rows)
    write_json(FEATURE_DICTIONARY_JSON, payload)
    write_markdown(FEATURE_DICTIONARY_MD, rows)
    print(f"Wrote {FEATURE_DICTIONARY_CSV}")
    print(f"Wrote {FEATURE_DICTIONARY_JSON}")
    print(f"Wrote {FEATURE_DICTIONARY_MD}")


if __name__ == "__main__":
    main()

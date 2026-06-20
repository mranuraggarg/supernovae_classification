"""PCA alignment trial for Phase 2 Tier 4 cross-survey transfer."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from phase2_tier1_benchmarks import RANDOM_STATE, VALIDATION_SPLIT, stratified_split_indices
from phase2_tier2_common import binary_metrics
from phase2_tier2_common import build_matrix, create_context
from phase2_tier4_common import (
    domain_splits_from_variants,
    load_variant_rows,
    save_json,
)
from phase2_tier3_model_compare import _fit_final_xgb, _select_best_xgb


RESULTS_DIR = "results/phase2_tier4_trial_pca"
SUMMARY_PATH = f"{RESULTS_DIR}/trial_pca_summary.md"
JSON_PATH = f"{RESULTS_DIR}/trial_pca_results.json"
N_COMPONENTS = 10


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def rows_to_xy(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = build_matrix(rows, feature_names)
    return x_values.astype(np.float32), y_values.astype(np.int32)


def standardize_with_train(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def pca_feature_names(n_components: int) -> list[str]:
    return [f"pca_{index + 1}" for index in range(n_components)]


def component_alignment_rows(pca_left: PCA, pca_right: PCA) -> list[dict[str, float]]:
    rows = []
    component_count = min(len(pca_left.components_), len(pca_right.components_))
    for index in range(component_count):
        left = pca_left.components_[index]
        right = pca_right.components_[index]
        cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
        rows.append(
            {
                "component": index + 1,
                "cosine_similarity": cosine,
                "absolute_cosine_similarity": abs(cosine),
                "spcc_explained_variance_ratio": float(pca_left.explained_variance_ratio_[index]),
                "plasticc_explained_variance_ratio": float(pca_right.explained_variance_ratio_[index]),
            }
        )
    return rows


def evaluate_with_projection(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    pca: PCA,
) -> dict[str, Any]:
    train_proj = pca.transform(train_x)
    test_proj = pca.transform(test_x)
    feature_names = pca_feature_names(train_proj.shape[1])
    train_idx, validation_idx = stratified_split_indices(train_y, VALIDATION_SPLIT, RANDOM_STATE)
    inner_train_x = train_proj[train_idx]
    inner_train_y = train_y[train_idx]
    validation_x = train_proj[validation_idx]
    validation_y = train_y[validation_idx]
    selection = _select_best_xgb(
        inner_train_x,
        inner_train_y,
        validation_x,
        validation_y,
        feature_names,
        seed=RANDOM_STATE,
    )
    _, probs = _fit_final_xgb(
        train_proj,
        train_y,
        test_proj,
        test_y,
        feature_names,
        seed=RANDOM_STATE,
        params=selection["params"],
        num_boost_round=selection["best_iteration"],
    )
    return {
        "model_name": "xgboost_pca",
        "selection": selection,
        "metrics": binary_metrics(test_y, probs),
    }


def write_summary(payload: dict[str, Any]) -> None:
    alignment_rows = payload["separate_pca"]["alignment_rows"]
    cross_projection = payload["cross_projection"]
    shared_pca = payload["shared_pca"]

    lines = [
        "# Phase 2 Tier 4 PCA Alignment Trial",
        "",
        "This trial evaluates whether SPCC and PLAsTiCC are separated mainly by a coordinate-system mismatch.",
        "",
        "## Separate PCA comparison",
        "",
        "| component | SPCC var | PLAsTiCC var | cosine | abs cosine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in alignment_rows:
        lines.append(
            f"| {row['component']} | {row['spcc_explained_variance_ratio']:.6f} | "
            f"{row['plasticc_explained_variance_ratio']:.6f} | {row['cosine_similarity']:.6f} | "
            f"{row['absolute_cosine_similarity']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Cross projection",
            f"- SPCC PCA, SPCC train -> SPCC test F1: {cross_projection['spcc_to_spcc']['metrics']['f1']:.6f}",
            f"- SPCC PCA, SPCC train -> PLAsTiCC test F1: {cross_projection['spcc_to_plasticc']['metrics']['f1']:.6f}",
            "",
            "## Shared PCA",
            f"- Shared PCA, SPCC train -> SPCC test F1: {shared_pca['spcc_to_spcc']['metrics']['f1']:.6f}",
            f"- Shared PCA, SPCC train -> PLAsTiCC test F1: {shared_pca['spcc_to_plasticc']['metrics']['f1']:.6f}",
        ]
    )

    with open(SUMMARY_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    splits = domain_splits_from_variants(context, variant_rows)
    if "plasticc" not in splits:
        raise FileNotFoundError("PLAsTiCC split is required for the PCA trial.")

    spcc_train_x, spcc_train_y = rows_to_xy(splits["spcc"]["trainval"], context.compact_features)
    spcc_test_x, spcc_test_y = rows_to_xy(splits["spcc"]["test"], context.compact_features)
    plasticc_train_x, plasticc_train_y = rows_to_xy(splits["plasticc"]["trainval"], context.compact_features)
    plasticc_test_x, plasticc_test_y = rows_to_xy(splits["plasticc"]["test"], context.compact_features)

    spcc_train_scaled, spcc_test_scaled, spcc_mean, spcc_std = standardize_with_train(spcc_train_x, spcc_test_x)
    plasticc_train_scaled, plasticc_test_scaled, _, _ = standardize_with_train(plasticc_train_x, plasticc_test_x)

    plasticc_test_scaled_on_spcc = (plasticc_test_x - spcc_mean) / spcc_std
    plasticc_train_scaled_on_spcc = (plasticc_train_x - spcc_mean) / spcc_std

    spcc_pca = PCA(n_components=min(N_COMPONENTS, spcc_train_scaled.shape[1]), random_state=42)
    plasticc_pca = PCA(n_components=min(N_COMPONENTS, plasticc_train_scaled.shape[1]), random_state=42)
    spcc_pca.fit(spcc_train_scaled)
    plasticc_pca.fit(plasticc_train_scaled)

    alignment_rows = component_alignment_rows(spcc_pca, plasticc_pca)

    cross_spcc_to_spcc = evaluate_with_projection(
        spcc_train_scaled,
        spcc_train_y,
        spcc_test_scaled,
        spcc_test_y,
        pca=spcc_pca,
    )
    cross_spcc_to_plasticc = evaluate_with_projection(
        spcc_train_scaled,
        spcc_train_y,
        plasticc_test_scaled_on_spcc,
        plasticc_test_y,
        pca=spcc_pca,
    )

    stacked_train = np.vstack([spcc_train_scaled, plasticc_train_scaled_on_spcc])
    shared_pca = PCA(n_components=min(N_COMPONENTS, stacked_train.shape[1]), random_state=42)
    shared_pca.fit(stacked_train)
    shared_spcc_to_spcc = evaluate_with_projection(
        spcc_train_scaled,
        spcc_train_y,
        spcc_test_scaled,
        spcc_test_y,
        pca=shared_pca,
    )
    shared_spcc_to_plasticc = evaluate_with_projection(
        spcc_train_scaled,
        spcc_train_y,
        plasticc_test_scaled_on_spcc,
        plasticc_test_y,
        pca=shared_pca,
    )

    payload = {
        "trial": "pca_alignment",
        "n_components": N_COMPONENTS,
        "separate_pca": {
            "alignment_rows": alignment_rows,
            "spcc_explained_variance_ratio": [float(value) for value in spcc_pca.explained_variance_ratio_],
            "plasticc_explained_variance_ratio": [float(value) for value in plasticc_pca.explained_variance_ratio_],
        },
        "cross_projection": {
            "spcc_to_spcc": cross_spcc_to_spcc,
            "spcc_to_plasticc": cross_spcc_to_plasticc,
        },
        "shared_pca": {
            "spcc_to_spcc": shared_spcc_to_spcc,
            "spcc_to_plasticc": shared_spcc_to_plasticc,
        },
    }
    save_json(JSON_PATH, payload)
    write_summary(payload)
    print(json.dumps({"json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()

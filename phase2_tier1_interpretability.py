"""Interpretability analysis for the saved Phase 2 Tier-1 compact baseline."""

from __future__ import annotations

import csv
import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


BASELINE_NAME = "phase2_tier1_compact_baseline"
RESULTS_DIR = "results/phase2_tier1"
PLOTS_DIR = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_plots")

CSV_PATH = "data/processed/phase2_tier1_compact_baseline.csv"
MODEL_PATH = "models/phase2_tier1/compact_baseline/phase2_tier1_compact_baseline_xgb.json"
METRICS_PATH = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_metrics.json")
PERMUTATION_PATH = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_importance.json")
MANIFEST_PATH = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_manifest.json")

RANDOM_STATE = 42
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2


PHYSICAL_MEANINGS = {
    "z_peak_flux": "Peak brightness scale in z band.",
    "r_mean_flux": "Mean brightness level in r band across the reconstructed light curve.",
    "peak_color_g_minus_r": "Color near peak from the g-to-r flux ratio.",
    "i_peak_flux": "Peak brightness scale in i band.",
    "peak_color_r_minus_i": "Color near peak from the r-to-i flux ratio.",
    "peak_color_i_minus_z": "Color near peak from the i-to-z flux ratio.",
    "g_mean_flux": "Mean brightness level in g band across the reconstructed light curve.",
    "r_peak_flux": "Peak brightness scale in r band.",
    "z_std_flux": "Variability or spread of reconstructed z-band flux values.",
    "i_amplitude": "Peak-to-trough amplitude in the i band.",
    "i_std_flux": "Variability or spread of reconstructed i-band flux values.",
    "time_span": "Observed temporal coverage of the transient.",
    "z_time_of_peak": "Time of peak in z band.",
    "i_time_of_peak": "Time of peak in i band.",
    "r_time_of_peak": "Time of peak in r band.",
    "r_std_flux": "Variability or spread of reconstructed r-band flux values.",
}

INTERPRETATIONS = {
    "z_peak_flux": "Band-specific peak brightness is a strong classifier signal, consistent with transient luminosity structure.",
    "r_mean_flux": "Average r-band brightness encodes scale information beyond a single peak snapshot.",
    "peak_color_g_minus_r": "Color near peak captures temperature or spectral-slope differences characteristic of Ia evolution.",
    "i_peak_flux": "i-band peak brightness contributes strongly to separating Ia from non-Ia events.",
    "peak_color_r_minus_i": "r-i color acts as a spectral-slope proxy near peak.",
    "peak_color_i_minus_z": "i-z color contributes to distinguishing redder or later-phase light-curve behavior.",
    "g_mean_flux": "Mean g-band brightness adds complementary information to peak-only brightness proxies.",
    "r_peak_flux": "r-band peak brightness remains informative even after compact pruning.",
    "z_std_flux": "Spread in z-band flux captures shape or variability around peak.",
    "i_amplitude": "i-band amplitude reflects how strongly the light curve rises and falls around peak.",
    "i_std_flux": "i-band variability helps capture light-curve shape, not just scale.",
    "time_span": "Observed temporal coverage contributes because different classes occupy different effective time windows.",
    "z_time_of_peak": "Inter-band timing helps capture peak-lag structure in the transient.",
    "i_time_of_peak": "Peak timing in i band contributes to the time-evolution signature of Ia light curves.",
    "r_time_of_peak": "Peak timing in r band reflects band-lag physics and phase structure.",
    "r_std_flux": "r-band spread adds another light-curve-shape cue after pruning redundant amplitudes.",
}


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {
                "snid": int(float(row["snid"])),
                "label_name": row["label_name"],
                "label_id": int(float(row["label_id"])),
                "sim_z": float(row["sim_z"]),
            }
            for key, value in row.items():
                if key in parsed:
                    continue
                if key in {"snid", "label_name", "label_id", "sim_z"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def stratified_split_indices(labels: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_size))
        test_count = min(max(test_count, 1), len(shuffled) - 1)
        test_indices.extend(shuffled[:test_count])
        train_indices.extend(shuffled[test_count:])
    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_values = train_x.mean(axis=0)
    std_values = train_x.std(axis=0)
    std_values[std_values == 0.0] = 1.0
    return (train_x - mean_values) / std_values, (other_x - mean_values) / std_values


def roc_auc_score_numpy(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(pos) * len(neg)))


def average_precision_numpy(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    positive_total = max(int(np.sum(y_true == 1)), 1)
    recall = tp_cumsum / positive_total
    ap = 0.0
    previous_recall = 0.0
    for p_value, r_value, label in zip(precision, recall, y_sorted):
        if label == 1:
            ap += p_value * (r_value - previous_recall)
            previous_recall = r_value
    return float(ap)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_numpy(y_true, probs),
        "pr_auc": average_precision_numpy(y_true, probs),
    }


def build_fixed_test_split(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, val_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    _ = val_idx  # validation is not used directly here, but the split is mirrored intentionally.

    trainval_rows = [rows[index] for index in trainval_idx]
    test_rows = [rows[index] for index in test_idx]
    x_trainval = np.array([[row[name] for name in feature_names] for row in trainval_rows], dtype=np.float32)
    x_test = np.array([[row[name] for name in feature_names] for row in test_rows], dtype=np.float32)
    y_test = np.array([1 if row["label_name"] == "Ia" else 0 for row in test_rows], dtype=np.int32)
    x_trainval_scaled, x_test_scaled = standardize(x_trainval, x_test)
    return x_test_scaled, y_test, np.array(test_rows, dtype=object)


def load_saved_metrics(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def load_manifest(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def load_permutation_rows(path: str) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)["permutation_importance"]


def shap_summary_rows(feature_names: list[str], shap_values: np.ndarray) -> list[dict]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "feature": feature_names[index],
                "mean_abs_shap": float(mean_abs[index]),
                "shap_rank": rank,
            }
        )
    return rows


def plot_summary_bar(summary_rows: list[dict], output_path: str) -> None:
    top_rows = summary_rows[:10]
    labels = [row["feature"] for row in reversed(top_rows)]
    values = [row["mean_abs_shap"] for row in reversed(top_rows)]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="#3b82f6")
    plt.xlabel("mean(|SHAP|)")
    plt.title("Compact Baseline SHAP Global Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_summary_beeswarm(feature_names: list[str], x_test: np.ndarray, shap_values: np.ndarray, output_path: str) -> None:
    top_indices = np.argsort(-np.mean(np.abs(shap_values), axis=0))[:10]
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap("coolwarm")
    for y_position, feature_index in enumerate(reversed(top_indices)):
        feature_values = x_test[:, feature_index]
        shap_column = shap_values[:, feature_index]
        order = np.argsort(shap_column)
        feature_values = feature_values[order]
        shap_column = shap_column[order]
        jitter = np.linspace(-0.25, 0.25, len(shap_column))
        normalized = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-9)
        plt.scatter(
            shap_column,
            np.full_like(shap_column, y_position, dtype=float) + jitter,
            c=normalized,
            cmap=cmap,
            s=8,
            alpha=0.5,
            edgecolors="none",
        )
    plt.yticks(range(len(top_indices)), [feature_names[index] for index in reversed(top_indices)])
    plt.xlabel("SHAP value")
    plt.title("Compact Baseline SHAP Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_dependence(feature_name: str, feature_values: np.ndarray, shap_column: np.ndarray, probs: np.ndarray, output_prefix: str) -> dict:
    dependence_path = f"{output_prefix}_{feature_name}_dependence.png"
    probability_path = f"{output_prefix}_{feature_name}_probability.png"

    plt.figure(figsize=(6, 4))
    plt.scatter(feature_values, shap_column, s=10, alpha=0.45, color="#2563eb")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel(feature_name)
    plt.ylabel("SHAP value")
    plt.title(f"{feature_name}: SHAP dependence")
    plt.tight_layout()
    plt.savefig(dependence_path, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(feature_values, probs, s=10, alpha=0.45, color="#dc2626")
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Ia probability")
    plt.title(f"{feature_name}: feature value vs Ia probability")
    plt.tight_layout()
    plt.savefig(probability_path, dpi=160)
    plt.close()

    corr = float(np.corrcoef(feature_values, shap_column)[0, 1]) if np.std(feature_values) > 0 and np.std(shap_column) > 0 else 0.0
    direction = "higher values increase Ia probability" if corr > 0.1 else "higher values decrease Ia probability" if corr < -0.1 else "effect is mixed or weakly monotonic"
    return {
        "feature": feature_name,
        "feature_shap_correlation": corr,
        "directional_interpretation": direction,
        "dependence_plot_path": dependence_path,
        "probability_plot_path": probability_path,
    }


def write_interpretation_table(rows: list[dict], output_csv: str, output_md: str) -> None:
    fieldnames = ["feature", "physical_meaning", "shap_rank", "mean_abs_shap", "mean_pr_auc_drop", "mean_f1_drop", "relative_importance", "directional_interpretation", "interpretation"]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| feature | physical meaning | SHAP rank | mean(|SHAP|) | perm PR-AUC drop | perm F1 drop | relative importance | interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature']} | {row['physical_meaning']} | {row['shap_rank']} | {row['mean_abs_shap']:.6f} | "
            f"{row['mean_pr_auc_drop']:.6f} | {row['mean_f1_drop']:.6f} | {row['relative_importance']} | {row['interpretation']} |"
        )
    with open(output_md, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    metrics_payload = load_saved_metrics(METRICS_PATH)
    manifest_payload = load_manifest(MANIFEST_PATH)
    feature_names = manifest_payload["feature_names"]

    rows = load_rows(CSV_PATH)
    x_test, y_test, test_rows = build_fixed_test_split(rows, feature_names)

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    probs = booster.predict(dtest)
    metrics = binary_metrics(y_test, probs)

    shap_contribs = booster.predict(dtest, pred_contribs=True)
    shap_values = shap_contribs[:, :-1]
    bias_term = shap_contribs[:, -1]

    summary_rows = shap_summary_rows(feature_names, shap_values)
    permutation_rows = {row["feature"]: row for row in load_permutation_rows(PERMUTATION_PATH)}
    top_features = [row["feature"] for row in summary_rows[:8]]

    summary_bar_path = os.path.join(PLOTS_DIR, f"{BASELINE_NAME}_shap_summary_bar.png")
    summary_beeswarm_path = os.path.join(PLOTS_DIR, f"{BASELINE_NAME}_shap_summary.png")
    plot_summary_bar(summary_rows, summary_bar_path)
    plot_summary_beeswarm(feature_names, x_test, shap_values, summary_beeswarm_path)

    directional_rows = {}
    feature_matrix = np.array([[row[name] for name in feature_names] for row in test_rows], dtype=np.float32)
    for feature_name in top_features:
        feature_index = feature_names.index(feature_name)
        directional_rows[feature_name] = plot_dependence(
            feature_name,
            feature_matrix[:, feature_index],
            shap_values[:, feature_index],
            probs,
            os.path.join(PLOTS_DIR, BASELINE_NAME),
        )

    max_mean_abs = max(row["mean_abs_shap"] for row in summary_rows)
    interpretation_rows = []
    for row in summary_rows:
        feature = row["feature"]
        mean_abs_shap = row["mean_abs_shap"]
        relative_importance = "dominant" if mean_abs_shap >= 0.7 * max_mean_abs else "strong" if mean_abs_shap >= 0.4 * max_mean_abs else "moderate" if mean_abs_shap >= 0.2 * max_mean_abs else "weaker"
        interpretation_rows.append(
            {
                "feature": feature,
                "physical_meaning": PHYSICAL_MEANINGS.get(feature, "Compact baseline feature."),
                "shap_rank": row["shap_rank"],
                "mean_abs_shap": mean_abs_shap,
                "mean_pr_auc_drop": permutation_rows[feature]["mean_pr_auc_drop"],
                "mean_f1_drop": permutation_rows[feature]["mean_f1_drop"],
                "relative_importance": relative_importance,
                "directional_interpretation": directional_rows.get(feature, {}).get("directional_interpretation", "See dependence plot."),
                "interpretation": INTERPRETATIONS.get(feature, "Feature contributes to compact-baseline discrimination."),
            }
        )

    table_csv = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_interpretation_table.csv")
    table_md = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_interpretation_table.md")
    write_interpretation_table(interpretation_rows, table_csv, table_md)

    payload = {
        "baseline_name": BASELINE_NAME,
        "model_path": MODEL_PATH,
        "reproduced_test_metrics": metrics,
        "saved_test_metrics": metrics_payload["test_metrics"],
        "metric_deltas_vs_saved": {
            key: metrics[key] - metrics_payload["test_metrics"][key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        },
        "shap_summary": summary_rows,
        "permutation_importance": [permutation_rows[row["feature"]] for row in summary_rows],
        "bias_term_mean": float(mean(bias_term.tolist())),
        "summary_bar_plot": summary_bar_path,
        "summary_plot": summary_beeswarm_path,
        "directional_analysis": [directional_rows[name] for name in top_features],
        "interpretation_table_csv": table_csv,
        "interpretation_table_markdown": table_md,
    }
    output_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_interpretability.json")
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"interpretability_path": output_path, "summary_plot": summary_beeswarm_path}, indent=2))


if __name__ == "__main__":
    main()

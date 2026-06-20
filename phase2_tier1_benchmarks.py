"""Benchmark feature-set comparisons for Phase 2 Tier 1 SPCC features."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


CSV_PATH = "data/processed/spcc_features_tier1.csv"
RESULTS_DIR = "results/phase2_tier1"
MODELS_DIR = "models/phase2_tier1/benchmarks"
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
EPOCHS = 120
LEARNING_RATE = 1e-2


SURVEY_CONTEXT_FEATURES = [
    "observation_count",
    "time_span",
    "total_snr",
]

PHOTOMETRIC_CORE_FEATURES = [
    "peak_flux_all",
    "amplitude_all",
    "mean_flux_all",
    "std_flux_all",
    "g_peak_flux",
    "g_mean_flux",
    "g_std_flux",
    "g_amplitude",
    "r_peak_flux",
    "r_mean_flux",
    "r_std_flux",
    "r_amplitude",
    "i_peak_flux",
    "i_mean_flux",
    "i_std_flux",
    "i_amplitude",
    "z_peak_flux",
    "z_mean_flux",
    "z_std_flux",
    "z_amplitude",
    "peak_color_g_minus_r",
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
]

PEAK_TIME_FEATURES = [
    "time_of_peak_all",
    "g_time_of_peak",
    "r_time_of_peak",
    "i_time_of_peak",
    "z_time_of_peak",
]

FEATURE_SETS = {
    "survey_context_only": SURVEY_CONTEXT_FEATURES,
    "photometric_core": PHOTOMETRIC_CORE_FEATURES,
    "full_tier1_baseline": PHOTOMETRIC_CORE_FEATURES + PEAK_TIME_FEATURES + SURVEY_CONTEXT_FEATURES,
}


@dataclass
class BenchmarkResult:
    name: str
    feature_count: int
    best_epoch: int
    validation_f1: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


class LogisticBaseline(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _device() -> torch.device:
    return torch.device("cpu")


def load_rows(csv_path: str = CSV_PATH) -> list[dict]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {"snid": int(row["snid"]), "label_name": row["label_name"]}
            for key, value in row.items():
                if key in {"snid", "label_name"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def build_matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([1.0 if row["label_name"] == "Ia" else 0.0 for row in rows], dtype=np.float32)
    return x, y


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


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    y_true_int = y_true.astype(np.int32)

    tp = int(np.sum((preds == 1) & (y_true_int == 1)))
    fp = int(np.sum((preds == 1) & (y_true_int == 0)))
    tn = int(np.sum((preds == 0) & (y_true_int == 0)))
    fn = int(np.sum((preds == 0) & (y_true_int == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true_int)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_numpy(y_true_int, probs),
        "pr_auc": average_precision_numpy(y_true_int, probs),
    }


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


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = EPOCHS,
) -> tuple[dict, int, float]:
    device = _device()
    model = LogisticBaseline(x_train.shape[1]).to(device)

    positive_count = float(np.sum(y_train == 1.0))
    negative_count = float(np.sum(y_train == 0.0))
    pos_weight_value = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(val_x)).detach().cpu().numpy()
        val_metrics = binary_metrics(y_val, val_probs)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return best_state, best_epoch, best_f1


def fit_for_epochs(x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> nn.Module:
    device = _device()
    model = LogisticBaseline(x_train.shape[1]).to(device)
    positive_count = float(np.sum(y_train == 1.0))
    negative_count = float(np.sum(y_train == 0.0))
    pos_weight_value = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    return model


def run_benchmarks() -> dict:
    rows = load_rows()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, val_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    train_rows = [rows[trainval_idx[index]] for index in train_idx]
    val_rows = [rows[trainval_idx[index]] for index in val_idx]
    trainval_rows = [rows[index] for index in trainval_idx]
    test_rows = [rows[index] for index in test_idx]

    results = []
    for name, feature_names in FEATURE_SETS.items():
        x_train_raw, y_train = build_matrix(train_rows, feature_names)
        x_val_raw, y_val = build_matrix(val_rows, feature_names)
        x_train_scaled, x_val_scaled, mean, std = standardize(x_train_raw, x_val_raw)

        _, best_epoch, best_val_f1 = train_model(x_train_scaled, y_train, x_val_scaled, y_val)

        x_trainval_raw, y_trainval = build_matrix(trainval_rows, feature_names)
        x_test_raw, y_test = build_matrix(test_rows, feature_names)
        x_trainval_scaled, x_test_scaled, final_mean, final_std = standardize(x_trainval_raw, x_test_raw)
        final_model = fit_for_epochs(x_trainval_scaled, y_trainval, best_epoch)

        device = _device()
        test_x = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
        final_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(final_model(test_x)).detach().cpu().numpy()
        metrics = binary_metrics(y_test, probs)

        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "feature_names": feature_names,
                "mean": final_mean,
                "std": final_std,
                "best_epoch": best_epoch,
            },
            os.path.join(MODELS_DIR, f"{name}.pt"),
        )

        results.append(
            BenchmarkResult(
                name=name,
                feature_count=len(feature_names),
                best_epoch=best_epoch,
                validation_f1=best_val_f1,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                roc_auc=metrics["roc_auc"],
                pr_auc=metrics["pr_auc"],
            )
        )

    split_manifest = {
        "random_state": RANDOM_STATE,
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "test_count": len(test_rows),
        "train_ia_count": int(sum(row["label_name"] == "Ia" for row in train_rows)),
        "validation_ia_count": int(sum(row["label_name"] == "Ia" for row in val_rows)),
        "test_ia_count": int(sum(row["label_name"] == "Ia" for row in test_rows)),
        "device": str(_device()),
    }

    output = {
        "task": "binary_Ia_vs_non_Ia_torch_logistic_benchmarks",
        "artifact": CSV_PATH,
        "feature_sets": FEATURE_SETS,
        "split_manifest": split_manifest,
        "results": [asdict(result) for result in results],
    }
    with open(os.path.join(RESULTS_DIR, "phase2_tier1_benchmark_results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    return output


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    print(json.dumps(benchmark_results, indent=2))

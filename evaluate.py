import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Fully synced directory structure (match your actual filesystem)
MODEL_PATHS = {
    "linear_without_SMOTE": {
        "model": "models/linear/linear_without_SMOTE.pt",
        "results": "models/linear/linear_without_SMOTE_results.json",
        "scaler": "models/linear/linear_without_SMOTE_scaler.pkl"
    },
    "linear_with_SMOTE": {
        "model": "models/linear/linear_with_SMOTE.pt",
        "results": "models/linear/linear_with_SMOTE_results.json",
        "scaler": "models/linear/linear_with_SMOTE_scaler.pkl"
    },
    "random_forest_without_SMOTE": {
        "model": "models/random_forest/without_SMOTE/model.pkl",
        "results": "models/random_forest/without_SMOTE/results.json",
        "scaler": "models/random_forest/without_SMOTE/scaler.pkl"
    },
    "random_forest_with_SMOTE": {
        "model": "models/random_forest/with_SMOTE/model.pkl",
        "results": "models/random_forest/with_SMOTE/results.json",
        "scaler": "models/random_forest/with_SMOTE/scaler.pkl"
    },
    "xgboost_without_SMOTE": {
        "model": "models/xgboost/without_SMOTE/model.pkl",
        "results": "models/xgboost/without_SMOTE/results.json",
        "scaler": "models/xgboost/without_SMOTE/scaler.pkl"
    },
    "xgboost_with_SMOTE": {
        "model": "models/xgboost/with_SMOTE/model.pkl",
        "results": "models/xgboost/with_SMOTE/results.json",
        "scaler": "models/xgboost/with_SMOTE/scaler.pkl"
    }
}

# Bootstrap confidence interval calculation
def bootstrap(model_func, X_test, Y_test, n_iterations=1000, random_state=42):
    np.random.seed(random_state)
    metrics_list = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_sample = X_test[indices]
        Y_sample = Y_test[indices]
        metrics = model_func(X_sample, Y_sample)
        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list, columns=['precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
    
    summary = pd.DataFrame({
        'mean': df.mean(),
        'lower': df.apply(lambda x: np.percentile(x, 2.5)),
        'upper': df.apply(lambda x: np.percentile(x, 97.5))
    })
    
    return summary

# Evaluation metrics calculation
def metrics(y_true, y_pred, y_probs):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_probs)
    p, r, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(r, p)
    return precision, recall, f1, roc_auc, pr_auc

# Linear model class (PyTorch)
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.linear(x)

# Linear evaluation
def evaluate_linear(model_path, scaler_path, X_test, Y_test):
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X_test)
    model = LinearClassifier(X_scaled.shape[1], 2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        probs = model(X_tensor).softmax(1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return metrics(Y_test, preds, probs)

# Random Forest evaluation
def evaluate_random_forest(model_path, scaler_path, X_test, Y_test):
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X_test)
    model = load(model_path)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return metrics(Y_test, preds, probs)

# XGBoost evaluation
def evaluate_xgboost(model_path, scaler_path, X_test, Y_test):
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X_test)
    # Updated XGBoost model definition with optimized hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=7, 
        learning_rate=0.04, 
        subsample=0.85, 
        colsample_bytree=0.8,
        gamma=0, 
        reg_alpha=0.01, 
        reg_lambda=1
    )
    model = load(model_path)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return metrics(Y_test, preds, probs)

# Master evaluation function
def evaluate_all(X_test, Y_test):
    results = {}
    X_test_flat = X_test.reshape(len(X_test), -1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    for model_key, paths in MODEL_PATHS.items():
        print(f"\n🚀 Evaluating: {model_key}")

        if "linear" in model_key:
            evaluator = lambda X, Y: evaluate_linear(paths['model'], paths['scaler'], X, np.argmax(Y, 1))
        elif "random_forest" in model_key:
            evaluator = lambda X, Y: evaluate_random_forest(paths['model'], paths['scaler'], X, np.argmax(Y, 1))
        elif "xgboost" in model_key:
            evaluator = lambda X, Y: evaluate_xgboost(paths['model'], paths['scaler'], X, np.argmax(Y, 1))

        bootstrap_summary = bootstrap(evaluator, X_test_flat, Y_test)
        print(bootstrap_summary)

        results[model_key] = bootstrap_summary

    return results
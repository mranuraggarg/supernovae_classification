import os
import json
import numpy as np
import optuna
import xgboost as xgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from dataset import load_saved_data
from sklearn.model_selection import StratifiedKFold

# ✅ Define Directories
SAVE_DIR = "models/xgboost"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Flatten & Normalize Data Function
def preprocess_data(X_train, X_test, Y_train):
    """
    Preprocess dataset: Flatten, scale, and handle labels.
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    Y_train_labels = np.argmax(Y_train, axis=1)
    
    return X_train_flat, X_test_flat, Y_train_labels

# ✅ Optuna Hyperparameter Tuning
def optimize_xgboost(X_train, Y_train, n_trials=50):
    """
    Uses Optuna to find the best hyperparameters for XGBoost.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
            "tree_method": "hist",
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train, Y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train_fold, Y_train_fold)

            Y_pred = model.predict(X_val_fold)
            f1_scores.append(f1_score(Y_val_fold, Y_pred, average="weighted"))

        return np.mean(f1_scores)

    # ✅ Run Optuna Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

# ✅ Train XGBoost Without SMOTE
def train_xgboost_no_smote(X_train, Y_train, X_test, Y_test):
    """
    Train and evaluate XGBoost model without SMOTE using Optuna best parameters.
    """

    # ✅ Preprocess Data
    X_train_flat, X_test_flat, Y_train_labels = preprocess_data(X_train, X_test, Y_train)

    # ✅ Scale Data
    scaler_no_SMOTE = StandardScaler()
    X_train_scaled = scaler_no_SMOTE.fit_transform(X_train_flat)
    X_test_scaled = scaler_no_SMOTE.transform(X_test_flat)
    dump(scaler_no_SMOTE, os.path.join(SAVE_DIR, "xgboost_scaler_without_SMOTE.pkl"))

    # ✅ Optimize Hyperparameters
    best_params = optimize_xgboost(X_train_scaled, Y_train_labels)

    # ✅ Train Model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_scaled, Y_train_labels)

    # ✅ Save Model
    model.save_model(os.path.join(SAVE_DIR, "xgboost_without_SMOTE.json"))

    # ✅ Convert Y_test from One-Hot to Single-Label Format
    Y_test_labels = np.argmax(Y_test, axis=1)

    # ✅ Evaluate Model
    Y_pred = model.predict(X_test_scaled)
    Y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]

    results = {
        "model": "xgboost_without_SMOTE",
        "precision": precision_score(Y_test_labels, Y_pred, average="weighted"),
        "recall": recall_score(Y_test_labels, Y_pred, average="weighted"),
        "f1_score": f1_score(Y_test_labels, Y_pred, average="weighted"),
        "roc_auc": roc_auc_score(Y_test_labels, Y_pred_probs),
        "best_params": best_params
    }
    # ✅ Save Results
    with open(os.path.join(SAVE_DIR, "xgboost_without_SMOTE_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ XGBoost Without SMOTE Training Completed.\n")

# ✅ Train XGBoost With SMOTE
def train_xgboost_with_smote(X_train, Y_train, X_test, Y_test):
    """
    Train and evaluate XGBoost model with SMOTE using Optuna best parameters.
    """

    # ✅ Preprocess Data
    X_train_flat, X_test_flat, Y_train_labels = preprocess_data(X_train, X_test, Y_train)

    # ✅ Apply SMOTE **Before Scaling**
    print("\n🚀 Applying SMOTE...")
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_SMOTE, Y_train_SMOTE = smote.fit_resample(X_train_flat, Y_train_labels)

    # ✅ Scale Data
    scaler_SMOTE = StandardScaler()
    X_train_scaled = scaler_SMOTE.fit_transform(X_train_SMOTE)
    X_test_scaled = scaler_SMOTE.transform(X_test_flat)
    dump(scaler_SMOTE, os.path.join(SAVE_DIR, "xgboost_scaler_with_SMOTE.pkl"))

    # ✅ Optimize Hyperparameters
    best_params = optimize_xgboost(X_train_scaled, Y_train_SMOTE)

    # ✅ Train Model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_scaled, Y_train_SMOTE)

    # ✅ Save Model
    model.save_model(os.path.join(SAVE_DIR, "xgboost_with_SMOTE.json"))

    # ✅ Convert Y_test from One-Hot to Single-Label Format
    Y_test_labels = np.argmax(Y_test, axis=1)

    # ✅ Evaluate Model
    Y_pred = model.predict(X_test_scaled)
    Y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]

    results = {
        "model": "xgboost_without_SMOTE",
        "precision": precision_score(Y_test_labels, Y_pred, average="weighted"),
        "recall": recall_score(Y_test_labels, Y_pred, average="weighted"),
        "f1_score": f1_score(Y_test_labels, Y_pred, average="weighted"),
        "roc_auc": roc_auc_score(Y_test_labels, Y_pred_probs),
        "best_params": best_params
    }

    # ✅ Save Results
    with open(os.path.join(SAVE_DIR, "xgboost_with_SMOTE_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ XGBoost With SMOTE Training Completed.\n")

def train_xgboost_optimized(X_train, Y_train, X_test, Y_test):
    """
    Trains an optimized XGBoost model using Optuna for hyperparameter tuning.
    - Saves the model, results, and scaler in `models/xgboost_optimized/`
    """

    # ✅ Ensure output directory exists
    SAVE_DIR = "models/xgboost_optimized"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ✅ Load Data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    Y_train_labels = np.argmax(Y_train, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # ✅ Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    dump(scaler, os.path.join(SAVE_DIR, "xgboost_scaler.pkl"))

    # ✅ Optuna Optimization with Stratified K-Fold CV
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
            "tree_method": "hist",  # ✅ Use "hist" (or "gpu_hist" for older XGBoost)
            "device": "cuda"  # ✅ Ensures GPU usage
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train_scaled, Y_train_labels):
            X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
            Y_train_fold, Y_val_fold = Y_train_labels[train_idx], Y_train_labels[val_idx]

            dtrain = xgb.DMatrix(X_train_fold, label=Y_train_fold)  # ✅ FIX: Removed `device="cuda"`
            dval = xgb.DMatrix(X_val_fold, label=Y_val_fold)  # ✅ FIX: Removed `device="cuda"`

            model = xgb.XGBClassifier(**params)
            model.fit(X_train_fold, Y_train_fold)

            Y_pred = model.predict(X_val_fold)
            f1_scores.append(f1_score(Y_val_fold, Y_pred, average="weighted"))

        return np.mean(f1_scores)

    # ✅ Run Optuna Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    # ✅ Train Final Model with Best Parameters
    best_params = study.best_params

    # ✅ Convert train & test sets to DMatrix
    dtrain = xgb.DMatrix(X_train_scaled, label=Y_train_labels)  # ✅ FIX: Removed `device="cuda"`
    dtest = xgb.DMatrix(X_test_scaled, label=Y_test_labels)  # ✅ FIX: Removed `device="cuda"`

    # ✅ Train Best Model
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train_scaled, Y_train_labels)

    # ✅ Save Model
    model_path = os.path.join(SAVE_DIR, "xgboost_optimized.json")
    best_model.save_model(model_path)
    print(f"✅ Best XGBoost Model Saved: {model_path}")

    # ✅ Evaluate Final Model
    Y_pred = best_model.predict(X_test_scaled)
    Y_pred_probs = best_model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(Y_test_labels, Y_pred, average="weighted")
    recall = recall_score(Y_test_labels, Y_pred, average="weighted")
    f1 = f1_score(Y_test_labels, Y_pred, average="weighted")
    roc_auc = roc_auc_score(Y_test_labels, Y_pred_probs)

    # ✅ Save Results
    results = {
        "model": "XGBoost Optimized",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_params": best_params
    }

    results_path = os.path.join(SAVE_DIR, "xgboost_optimized_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results Saved: {results_path}")

    # ✅ Print Final Model Performance
    print("\n📊 **Optimized XGBoost Performance**")
    df_results = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
        "XGBoost Optimized": [precision, recall, f1, roc_auc]
    })
    print(df_results.to_markdown())

# ✅ Run Training
if __name__ == "__main__":
    # ✅ Load the dataset
    X_train, Y_train, X_test, Y_test, *_ = load_saved_data(format="npz")

    # ✅ Train both models
    train_xgboost_no_smote(X_train, Y_train, X_test, Y_test)
    train_xgboost_with_smote(X_train, Y_train, X_test, Y_test)
    # train_xgboost_optimized(X_train, Y_train, X_test, Y_test)
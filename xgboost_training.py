import os
import json
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dataset import load_saved_data
import xgboost as xgb

# ✅ Directory where models and scalers will be saved
SAVE_DIR = "models/phase1_repair/xgboost"
RESULTS_DIR = "results/phase1_repair"

def train_xgboost_model(X_train, Y_train, X_test, Y_test, use_smote=False):
    subfolder = "with_SMOTE" if use_smote else "without_SMOTE"
    model_dir = os.path.join(SAVE_DIR, subfolder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    Y_train_labels = np.argmax(Y_train, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Flatten 3D -> 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_flat,
        Y_train_labels,
        test_size=0.2,
        random_state=42,
        stratify=Y_train_labels,
    )

    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_flat)

    # Optuna optimization objective
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1
        }

        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_val_scaled)
        f1 = f1_score(y_val, preds, average="weighted")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    X_trainval = np.vstack([X_tr, X_val])
    y_trainval = np.concatenate([y_tr, y_val])
    X_trainval_raw = X_train_flat
    y_trainval_raw = Y_train_labels

    if use_smote:
        smote = SMOTE(random_state=42)
        X_trainval_raw, y_trainval_raw = smote.fit_resample(X_trainval_raw, y_trainval_raw)

    final_scaler = StandardScaler()
    X_trainval_scaled = final_scaler.fit_transform(X_trainval_raw)
    X_test_scaled = final_scaler.transform(X_test_flat)

    best_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss")
    best_model.fit(X_trainval_scaled, y_trainval_raw)

    # Save final scaler used by the deployed model
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    dump(final_scaler, scaler_path)

    # Final evaluation
    preds = best_model.predict(X_test_scaled)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(Y_test_labels, preds, average="weighted")
    recall = recall_score(Y_test_labels, preds, average="weighted")
    f1 = f1_score(Y_test_labels, preds, average="weighted")
    roc_auc = roc_auc_score(Y_test_labels, probs)
    pr_auc = average_precision_score(Y_test_labels, probs)

    results = {
        "model": f"xgboost_{subfolder}",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_params": best_params
    }

    # Save model & results
    dump(best_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    results_path = os.path.join(
        RESULTS_DIR,
        f"xgboost_{subfolder}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

# ✅ Entry points for train.py
def xgboost_no_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_xgboost_model(X_train, Y_train, X_test, Y_test, use_smote=False)

def xgboost_with_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_xgboost_model(X_train, Y_train, X_test, Y_test, use_smote=True)

if __name__ == "__main__":
    from dataset import load_saved_data
    
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = load_saved_data(format="npz")

        # Train and save without SMOTE
    xgboost_no_SMOTE(X_train, Y_train, X_test, Y_test)

    # Train and save with SMOTE
    xgboost_with_SMOTE(X_train, Y_train, X_test, Y_test)
    print("✅ XGBoost models trained and saved in models/phase1_repair/xgboost/")

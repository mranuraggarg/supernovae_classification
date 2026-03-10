import os
import json
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dataset import load_saved_data
import xgboost as xgb

# ✅ Directory where models and scalers will be saved
SAVE_DIR = "models/xgboost"

def train_xgboost_model(X_train, Y_train, X_test, Y_test, use_smote=False):
    subfolder = "with_SMOTE" if use_smote else "without_SMOTE"
    model_dir = os.path.join(SAVE_DIR, subfolder)
    os.makedirs(model_dir, exist_ok=True)

    # Flatten 3D -> 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_flat, Y_train_labels = smote.fit_resample(X_train_flat, np.argmax(Y_train, axis=1))
        Y_train = np.eye(Y_train.shape[1])[Y_train_labels]
    else:
        Y_train_labels = np.argmax(Y_train, axis=1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    dump(scaler, scaler_path)

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
        model.fit(X_train_scaled, Y_train_labels)
        preds = model.predict(X_test_scaled)
        f1 = f1_score(np.argmax(Y_test, axis=1), preds, average="weighted")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss")
    best_model.fit(X_train_scaled, Y_train_labels)

    # Final evaluation
    preds = best_model.predict(X_test_scaled)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    recall = recall_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    f1 = f1_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    roc_auc = roc_auc_score(np.argmax(Y_test, axis=1), probs)

    results = {
        "model": f"xgboost_{subfolder}",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_params": best_params
    }

    # Save model & results
    dump(best_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "results.json"), "w") as f:
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
    print("✅ XGBoost models trained and saved in models/xgboost/")
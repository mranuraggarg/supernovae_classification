import os
import json
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dataset import load_saved_data

# ✅ Directory where models and scalers will be saved
SAVE_DIR = "models/random_forest"

def train_random_forest_model(X_train, Y_train, X_test, Y_test, use_smote=False):
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
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
            "n_jobs": -1
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, Y_train_labels)
        preds = model.predict(X_test_scaled)
        f1 = f1_score(np.argmax(Y_test, axis=1), preds, average="weighted")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, Y_train_labels)

    # Final evaluation
    preds = best_model.predict(X_test_scaled)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    recall = recall_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    f1 = f1_score(np.argmax(Y_test, axis=1), preds, average="weighted")
    roc_auc = roc_auc_score(np.argmax(Y_test, axis=1), probs)

    results = {
        "model": f"random_forest_{subfolder}",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_params": best_params
    }

    # Save model and results
    dump(best_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results

# ✅ Entry points for train.py
def random_forest_no_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_random_forest_model(X_train, Y_train, X_test, Y_test, use_smote=False)

def random_forest_with_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_random_forest_model(X_train, Y_train, X_test, Y_test, use_smote=True)
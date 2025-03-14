import os
import json
import numpy as np
import pandas as pd
import optuna
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dataset import load_saved_data  # ✅ Ensure this is correctly implemented

def random_forest_classification(X_train, Y_train, X_test, Y_test, use_smote=False, optim_trials=50):
    """
    Train an optimized Random Forest model using Optuna for hyperparameter tuning.
    - Supports **SMOTE-based balancing** when `use_smote=True`.
    - Saves **model & results** in `models/random_forest/`.
    """

    # ✅ Define model name & save directory
    model_name = "random_forest_with_SMOTE" if use_smote else "random_forest_without_SMOTE"
    save_dir = f"models/random_forest"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Flatten data (if 3D)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # ✅ Apply SMOTE if enabled
    if use_smote:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_train_flat, Y_train_labels = smote.fit_resample(X_train_flat, np.argmax(Y_train, axis=1))
        Y_train = np.eye(Y_train.shape[1])[Y_train_labels]  # Convert back to one-hot encoding
    else:
        Y_train_labels = np.argmax(Y_train, axis=1)

    # ✅ Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # ✅ Save the scaler for future use
    scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
    dump(scaler, scaler_path)

    # ✅ Define Optuna Objective Function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1  # ✅ Use all CPU cores for efficiency
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, Y_train_labels)

        # ✅ Predictions
        Y_pred = model.predict(X_test_scaled)
        Y_probs = model.predict_proba(X_test_scaled)[:, 1]

        # ✅ Compute Metrics
        f1 = f1_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
        return f1  # ✅ Optimize for F1-score

    # ✅ Run Optuna Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optim_trials)

    # ✅ Train the Best Model
    best_params = study.best_params
    best_rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf_model.fit(X_train_scaled, Y_train_labels)

    # ✅ Make Final Predictions
    Y_pred = best_rf_model.predict(X_test_scaled)
    Y_probs = best_rf_model.predict_proba(X_test_scaled)[:, 1]

    # ✅ Compute Final Metrics
    precision = precision_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    recall = recall_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    f1 = f1_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    roc_auc = roc_auc_score(np.argmax(Y_test, axis=1), Y_probs)

    # ✅ Save model & results
    save_rf_model_and_results(best_rf_model, {
        "model": "Random Forest",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_params": best_params
    }, save_dir, model_name)

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_params": best_params
    }

def save_rf_model_and_results(model, results, save_dir, model_name):
    """
    Saves the trained Random Forest model and results in structured files.
    - Model is saved as a `.pkl` file.
    - Results are saved as a `.json` file.
    """

    # ✅ Save the Random Forest Model
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")

    # ✅ Save the results as JSON
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved at: {results_path}")

if __name__ == "__main__":
    # ✅ Load dataset
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = load_saved_data(format="npz")

    # ✅ Train both models (with and without SMOTE)
    rf_results_without_smote = random_forest_classification(X_train, Y_train, X_test, Y_test, use_smote=False)
    rf_results_with_smote = random_forest_classification(X_train, Y_train, X_test, Y_test, use_smote=True)

    print("\n📊 **Final Results**")
    print(pd.DataFrame([rf_results_without_smote, rf_results_with_smote]).to_markdown(index=False))
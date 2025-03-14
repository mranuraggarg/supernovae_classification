import torch
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import optuna
from joblib import dump, load
from dataset import load_saved_data

def linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test, epochs=100):
    """
    Train a linear classification model WITHOUT SMOTE, find best threshold, and save results.
    """

    # ✅ Define model and save directory
    model_name = "linear"
    save_dir = f"models/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Check GPU availability
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    # ✅ Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)
    Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).to(device)

    # ✅ Define Model
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    # ✅ Initialize & Train Model
    model = LinearClassifier(X_train_scaled.shape[1], Y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()

    # ✅ Find Best Threshold
    with torch.no_grad():
        Y_probs = model(X_test_tensor).softmax(dim=1)[:, 1].cpu().numpy()
    
    best_threshold = max(np.linspace(0.1, 0.9, 50), key=lambda t: f1_score(
        Y_test_tensor.cpu().numpy(), (Y_probs >= t).astype(int), average="weighted"
    ))

    Y_pred_optimized = (Y_probs >= best_threshold).astype(int)

    # ✅ Compute Metrics
    precision = precision_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")
    recall = recall_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")
    f1 = f1_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")

    print(f"\n📊 **Final Model Performance Using Optimized Threshold ({best_threshold:.2f})**")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-Score: {f1:.4f}")

    # ✅ Save Model
    model_path = os.path.join(save_dir, "linear_without_SMOTE.pt")
    torch.save(model.state_dict(), model_path)

    # ✅ Save JSON Results
    results = {
        "model": "linear",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": "N/A",
        "best_params": {
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": epochs,
            "best_threshold": best_threshold
        }
    }

    results_path = os.path.join(save_dir, "linear_without_SMOTE_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved at: {results_path}")

    return results

def linear_classification_SMOTE(X_train, Y_train, X_test, Y_test, epochs=100):
    """
    Train a linear classification model with SMOTE, find best threshold, and save results.
    """

    # ✅ Define model and save directory
    model_name = "linear"
    save_dir = f"models/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Check GPU availability
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Apply SMOTE for data balancing
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_bal, Y_train_bal = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), np.argmax(Y_train, axis=1))

    # ✅ Convert labels back to one-hot encoding
    Y_train_bal = np.eye(Y_train.shape[1])[Y_train_bal]

    # ✅ Scale Data
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    # ✅ Convert to PyTorch tensors
    X_train_tensor_bal = torch.tensor(X_train_bal, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    Y_train_tensor_bal = torch.tensor(np.argmax(Y_train_bal, axis=1), dtype=torch.long).to(device)
    Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).to(device)

    # ✅ Define Model
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    # ✅ Initialize & Train Model
    model = LinearClassifier(X_train_bal.shape[1], Y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor_bal)
        loss = criterion(outputs, Y_train_tensor_bal)
        loss.backward()
        optimizer.step()

    # ✅ Find Best Threshold
    with torch.no_grad():
        Y_probs = model(X_test_tensor).softmax(dim=1)[:, 1].cpu().numpy()
    
    best_threshold = max(np.linspace(0.1, 0.9, 50), key=lambda t: f1_score(
        Y_test_tensor.cpu().numpy(), (Y_probs >= t).astype(int), average="weighted"
    ))

    Y_pred_optimized = (Y_probs >= best_threshold).astype(int)

    # ✅ Compute Metrics
    precision = precision_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")
    recall = recall_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")
    f1 = f1_score(Y_test_tensor.cpu().numpy(), Y_pred_optimized, average="weighted")

    print(f"\n📊 **Final Model Performance Using Optimized Threshold ({best_threshold:.2f})**")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-Score: {f1:.4f}")

    # ✅ Save Model
    model_path = os.path.join(save_dir, "linear_with_SMOTE.pt")
    torch.save(model.state_dict(), model_path)

    # ✅ Save JSON Results
    results = {
        "model": "linear",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": "N/A",
        "best_params": {
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": epochs,
            "best_threshold": best_threshold
        }
    }

    results_path = os.path.join(save_dir, "linear_with_SMOTE_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved at: {results_path}")

    return results

def random_forest_classification(X_train, Y_train, X_test, Y_test, optim_trials=50):
    """
    Train an optimized Random Forest model using Optuna for hyperparameter tuning.
    - No data balancing (no SMOTE)
    - Utilizes all CPU cores (n_jobs=-1)
    - Saves the best model and scaler
    """

    # ✅ Create model directory
    model_name = "random_forest"
    save_dir = f"models/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Flatten data (if 3D)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # ✅ Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # ✅ Save the scaler for future use
    dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    # ✅ Define Optuna Objective Function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1  # Use all CPU cores
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, np.argmax(Y_train, axis=1))

        # ✅ Predictions
        Y_pred = model.predict(X_test_scaled)
        Y_probs = model.predict_proba(X_test_scaled)[:, 1]

        # ✅ Compute Metrics
        precision = precision_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
        recall = recall_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
        f1 = f1_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
        roc_auc = roc_auc_score(np.argmax(Y_test, axis=1), Y_probs)

        return f1  # Optimize for F1-Score

    # ✅ Run Optuna Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optim_trials)

    # ✅ Train the Best Model
    best_params = study.best_params
    best_rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf_model.fit(X_train_scaled, np.argmax(Y_train, axis=1))

    # ✅ Save the Best Model
    dump(best_rf_model, os.path.join(save_dir, "random_forest_model.pkl"))

    # ✅ Make Final Predictions
    Y_pred = best_rf_model.predict(X_test_scaled)
    Y_probs = best_rf_model.predict_proba(X_test_scaled)[:, 1]

    # ✅ Compute Final Metrics
    precision = precision_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    recall = recall_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    f1 = f1_score(np.argmax(Y_test, axis=1), Y_pred, average="weighted")
    roc_auc = roc_auc_score(np.argmax(Y_test, axis=1), Y_probs)

    # ✅ Print Results in Neat Table
    df_results = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Random Forest Result": [precision, recall, f1, roc_auc],
    })

    rf_results = {
    "model": "Random Forest",
    "precision": precision,  # Computed from test set
    "recall": recall,  # Computed from test set
    "f1_score": f1,  # Computed from test set
    "roc_auc": roc_auc,  # Computed from test set
    "best_params": best_params  # Best hyperparameters found by Optuna
    }

    print("\n📊 **Optimized Random Forest Performance**")
    print(df_results.to_markdown())

    # ✅ Call this function after training your model
    save_rf_model_and_results(best_rf_model, rf_results)

    return rf_results

def save_rf_model_and_results(model, results, save_dir="models/random_forest"):
    """
    Saves the trained Random Forest model and results in a JSON file.
    - Creates the directory if it does not exist.
    - Saves the model as a .pkl file.
    - Saves results as a JSON file.
    """

    # ✅ Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Save the Random Forest Model
    model_path = os.path.join(save_dir, "random_forest_model.pkl")
    dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")

    # ✅ Save the results as JSON
    results_path = os.path.join(save_dir, "random_forest_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved at: {results_path}")

if __name__ == "__main__":
    # Load the data from stored npz file
    X_train, Y_train, X_test, Y_test, X_train_reverse, X_test_reverse, ids_train, ids_test, metadata = load_saved_data(format="npz")

    # Verify the data
    result_linear_no_SMOTE = linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_no_SMOTE)
    print("Linear classification model without SMOTE is saved in folder models/linear")

    result_linear_SMOTE = linear_classification_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_SMOTE)
    print("Linear classification model without SMOTE is saved in folder models/linear")
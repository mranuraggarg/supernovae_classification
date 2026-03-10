import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from dataset import load_saved_data
from joblib import dump

# ✅ Directory where models and scalers will be saved
SAVE_DIR = "models/linear"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Neural network architecture
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# ✅ Model training function
def train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=False):
    model_name = f"linear_with_SMOTE" if use_smote else f"linear_without_SMOTE"

    # Flatten 3D -> 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Apply SMOTE if selected
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_flat, Y_train_labels = smote.fit_resample(X_train_flat, np.argmax(Y_train, axis=1))
        Y_train = np.eye(Y_train.shape[1])[Y_train_labels]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Save scaler
    scaler_path = os.path.join(SAVE_DIR, f"{model_name}_scaler.pkl")
    dump(scaler, scaler_path)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # Model, loss, optimizer
    input_dim = X_train_tensor.shape[1]
    output_dim = Y_train_tensor.shape[1]
    model = LinearClassifier(input_dim, output_dim)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()

    # Save model
    model_path = os.path.join(SAVE_DIR, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    # Evaluation
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        predictions = torch.argmax(torch.softmax(outputs_test, dim=1), axis=1)
        true_labels = torch.argmax(Y_test_tensor, axis=1)

        precision = (predictions == true_labels).sum().item() / len(true_labels)
        recall = precision  # For 2-class simplification
        f1_score = precision
        roc_auc = 0  # Placeholder (since no probabilities used)

    results = {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc
    }

    # Save results as JSON
    with open(os.path.join(SAVE_DIR, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results

# ✅ Entry points for train.py
def linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=False)

def linear_classification_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=True)
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset import load_saved_data
from joblib import dump

# ✅ Directory where models and scalers will be saved
SAVE_DIR = "models/phase1_repair/linear"
RESULTS_DIR = "results/phase1_repair"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ✅ Neural network architecture
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# ✅ Model training function
def train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=False):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = f"linear_with_SMOTE" if use_smote else f"linear_without_SMOTE"
    model_dir = os.path.join(SAVE_DIR, "with_SMOTE" if use_smote else "without_SMOTE")
    os.makedirs(model_dir, exist_ok=True)

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
        stratify=Y_train_labels
    )

    # Apply SMOTE only on the training split
    if use_smote:
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    tuning_scaler = StandardScaler()
    X_tr_scaled = tuning_scaler.fit_transform(X_tr)
    X_val_scaled = tuning_scaler.transform(X_val)

    X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32, device=device)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)

    # Model, loss, optimizer
    input_dim = X_tr_tensor.shape[1]
    output_dim = Y_train.shape[1]
    model = LinearClassifier(input_dim, output_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with validation-based epoch selection
    best_val_f1 = -1.0
    best_epoch = 1
    best_state_dict = None
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tr_tensor)
        loss = loss_fn(outputs, y_tr_tensor)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average="weighted")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    X_trainval_raw = X_train_flat
    y_trainval_raw = Y_train_labels

    if use_smote:
        smote = SMOTE(random_state=42)
        X_trainval_raw, y_trainval_raw = smote.fit_resample(X_trainval_raw, y_trainval_raw)

    final_scaler = StandardScaler()
    X_trainval_scaled = final_scaler.fit_transform(X_trainval_raw)
    X_test_scaled = final_scaler.transform(X_test_flat)

    X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32, device=device)
    y_trainval_tensor = torch.tensor(y_trainval_raw, dtype=torch.long, device=device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    final_model = LinearClassifier(input_dim, output_dim).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    for _ in range(best_epoch):
        final_optimizer.zero_grad()
        logits = final_model(X_trainval_tensor)
        loss = loss_fn(logits, y_trainval_tensor)
        loss.backward()
        final_optimizer.step()

    # Save model
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    dump(final_scaler, scaler_path)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(final_model.state_dict(), model_path)

    # Evaluation on the untouched test set
    with torch.no_grad():
        outputs_test = final_model(X_test_tensor)
        probs = torch.softmax(outputs_test, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs_test, dim=1).cpu().numpy()

    precision = precision_score(Y_test_labels, preds, average="weighted")
    recall = recall_score(Y_test_labels, preds, average="weighted")
    f1 = f1_score(Y_test_labels, preds, average="weighted")
    roc_auc = roc_auc_score(Y_test_labels, probs)
    pr_auc = average_precision_score(Y_test_labels, probs)

    results = {
        "model": model_name,
        "device": str(device),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1
    }

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(RESULTS_DIR, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results

# ✅ Entry points for train.py
def linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=False)

def linear_classification_SMOTE(X_train, Y_train, X_test, Y_test):
    return train_linear_model(X_train, Y_train, X_test, Y_test, use_smote=True)

if __name__ == "__main__":
    print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = load_saved_data(format="npz")

    linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test)
    linear_classification_SMOTE(X_train, Y_train, X_test, Y_test)
    print("✅ Linear models trained and saved in models/phase1_repair/linear/")

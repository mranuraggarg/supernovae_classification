import os
import json
import numpy as np
import torch
import pandas as pd
import xgboost as xgb
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score
from torch import nn

# Define SPCC F1 score
def f1_score(y_true, y_test)
    '''
    Calculating SPCC F1-score
    = (1/ (TP + FN))x (TP^2 / (TP + 3 x FP))

    TP: True Positive
    FN: False Negative
    FP: False Positive
    '''





# ✅ Load Best Hyperparameters from JSON Results
def load_best_params(result_dirs):
    best_params_dict = {}
    for model_name, result_path in result_dirs.items():
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                saved_result = json.load(f)
                best_params_dict[model_name] = saved_result.get("best_params", {})
        else:
            print(f"⚠️ {result_path} not found! Skipping best parameters.")

    return best_params_dict

# ✅ Main Evaluation Function
def evaluate(X_test, Y_test):
    # ✅ Define Paths for Models & Results
    model_dirs = {
        "linear_without_SMOTE": "models/linear/linear_without_SMOTE.pt",
        "linear_with_SMOTE": "models/linear/linear_with_SMOTE.pt",
        "xgboost_without_SMOTE": "models/xgboost/xgboost_without_SMOTE.json",
        "xgboost_with_SMOTE": "models/xgboost/xgboost_with_SMOTE.json",
        "xgboost_optimized": "models/xgboost_optimized/xgboost_optimized.json",
        "random_forest_without_SMOTE": "models/random_forest/random_forest_without_SMOTE.pkl",
        "random_forest_with_SMOTE": "models/random_forest/random_forest_with_SMOTE.pkl"
    }

    result_dirs = {
        "linear_without_SMOTE": "models/linear/linear_without_SMOTE_results.json",
        "linear_with_SMOTE": "models/linear/linear_SMOTE_results.json",
        "xgboost_without_SMOTE": "models/xgboost/xgboost_without_SMOTE_results.json",
        "xgboost_with_SMOTE": "models/xgboost/xgboost_with_SMOTE_results.json",
        "xgboost_optimized": "models/xgboost_optimized/xgboost_optimized_results.json",
        "random_forest_without_SMOTE": "models/random_forest/random_forest_without_SMOTE_results.json",
        "random_forest_with_SMOTE": "models/random_forest/random_forest_with_SMOTE_results.json"
    }

    scaler_paths = {
        "xgboost_without_SMOTE": "models/xgboost/xgboost_scaler_without_SMOTE.pkl",
        "xgboost_with_SMOTE": "models/xgboost/xgboost_scaler_with_SMOTE.pkl",
        "xgboost_optimized": "models/xgboost_optimized/xgboost_scaler.pkl",
        "random_forest_without_SMOTE": "models/random_forest/random_forest_without_SMOTE_scaler.pkl",
        "random_forest_with_SMOTE": "models/random_forest/random_forest_with_SMOTE_scaler.pkl"
    }

    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten for consistency
    Y_test_labels = np.argmax(Y_test, axis=1)

    # ✅ Load Scalers
    scalers = {}
    for model_name, scaler_path in scaler_paths.items():
        if os.path.exists(scaler_path):
            scalers[model_name] = load(scaler_path)
            print(f"✅ Loaded Scaler for {model_name} from: {scaler_path}")
        else:
            print(f"⚠️ Scaler for {model_name} not found. Proceeding without scaling.")

    # ✅ Scale X_test for applicable models
    X_test_scaled = {model: scalers[model].transform(X_test_flat) if model in scalers else X_test_flat for model in scalers}

    # ✅ Load Linear Model Class
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    # ✅ Load and Evaluate Linear Models
    def evaluate_linear_model(model_path, X_test, Y_test):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        model = LinearClassifier(X_test.shape[1], Y_test.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Convert to PyTorch Tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).to(device)

        with torch.no_grad():
            Y_probs = model(X_test_tensor).softmax(dim=1)[:, 1].cpu().numpy()
            Y_pred = (Y_probs >= 0.5).astype(int)

        # Compute Metrics
        p, r, thresholds = precision_recall_curve(Y_test_tensor.cpu().numpy(), Y_probs)
        return {
            "precision": precision_score(Y_test_tensor.cpu().numpy(), Y_pred, average="weighted"),
            "recall": recall_score(Y_test_tensor.cpu().numpy(), Y_pred, average="weighted"),
            "f1_score": f1_score(Y_test_tensor.cpu().numpy(), Y_pred, average="weighted"),
            "roc_auc": roc_auc_score(Y_test_tensor.cpu().numpy(), Y_probs),
            "pr_auc": auc(r, p)
        }

    # ✅ Load and Evaluate XGBoost Models
    def evaluate_xgboost_model(model_path, X_test, Y_test):
        model = xgb.Booster()
        model.load_model(model_path)

        # Convert test data to DMatrix
        dtest = xgb.DMatrix(X_test)
        
        # Predict probabilities
        Y_pred_probs = model.predict(dtest)

        # Convert Probabilities to Class Labels
        Y_pred = (Y_pred_probs >= 0.5).astype(int)

        # Compute Metrics
        p, r, thresholds = precision_recall_curve(Y_test, Y_pred_probs)
        return {
            "precision": precision_score(Y_test, Y_pred, average="weighted"),
            "recall": recall_score(Y_test, Y_pred, average="weighted"),
            "f1_score": f1_score(Y_test, Y_pred, average="weighted"),
            "roc_auc": roc_auc_score(Y_test, Y_pred_probs),
            "pr_auc": auc(r, p)
        }

    # ✅ Load and Evaluate Random Forest Models
    def evaluate_random_forest_model(model_path, X_test, Y_test):
        model = load(model_path)
        Y_pred = model.predict(X_test)
        Y_probs = model.predict_proba(X_test)[:, 1]

        # Compute Metrics
        p, r, thresholds = precision_recall_curve(Y_test, Y_probs)
        return {
            "precision": precision_score(Y_test, Y_pred, average="weighted"),
            "recall": recall_score(Y_test, Y_pred, average="weighted"),
            "f1_score": f1_score(Y_test, Y_pred, average="weighted"),
            "roc_auc": roc_auc_score(Y_test, Y_probs),
            "pr_auc": auc(r, p)
        }

    # ✅ Gather All Model Results
    all_results = {}

    for model_name, model_path in model_dirs.items():
        print(f"\n🚀 Evaluating {model_name}...")

        if "linear" in model_name:
            all_results[model_name] = evaluate_linear_model(model_path, X_test_flat, Y_test)
        elif "xgboost" in model_name:
            # ✅ Identify correct scaler for each XGBoost variant
            if "optimized" in model_name:
                scaler_key = "xgboost_optimized"
            elif "with_SMOTE" in model_name:
                scaler_key = "xgboost_with_SMOTE"
            else:
                scaler_key = "xgboost_without_SMOTE"
            
            scaled_X_test = X_test_scaled.get(scaler_key, X_test_flat)  # Use the correct scaler if available
            all_results[model_name] = evaluate_xgboost_model(model_path, scaled_X_test, Y_test_labels)
        elif "random_forest" in model_name:
            scaler_key = "random_forest_with_SMOTE" if "with_SMOTE" in model_name else "random_forest_without_SMOTE"
            scaled_X_test = X_test_scaled.get(scaler_key, X_test_flat)  # Use the correct scaler if available
            all_results[model_name] = evaluate_random_forest_model(model_path, scaled_X_test, Y_test_labels)

    # ✅ Load Best Hyperparameters
    best_params_dict = load_best_params(result_dirs)

    # ✅ Add Best Parameters to Results
    for model_name in all_results:
        all_results[model_name]["best_params"] = best_params_dict.get(model_name, "N/A")

    # ✅ Create Final Comparison Table
    df_final_results = pd.DataFrame.from_dict(all_results, orient="index")
    df_final_results.reset_index(inplace=True)
    df_final_results.rename(columns={"index": "Model"}, inplace=True)

    import textwrap

    # ✅ Function to format best_params with word wrapping
    def format_best_params(params, width=60):
        if isinstance(params, dict):
            formatted_params = ", ".join([f"{k}: {round(v, 6) if isinstance(v, float) else v}" for k, v in params.items()])
            return "\n".join(textwrap.wrap(formatted_params, width))  # Wrap text
        return params

    # ✅ Apply formatting
    df_final_results["best_params"] = df_final_results["best_params"].apply(lambda x: format_best_params(x, width=50))

    # ✅ Adjust display settings
    pd.set_option("display.max_colwidth", None)  # Ensure full text visibility
    pd.set_option("display.float_format", "{:.4f}".format)  # Format floats for uniformity

    # ✅ Print Final Table with improved formatting
    print("\n📊 **Final Model Comparison Table**\n")
    print(df_final_results.to_markdown(index=False, tablefmt="grid"))  # `grid` improves readability

    # ✅ Return Final Results as Dictionary
    return all_results
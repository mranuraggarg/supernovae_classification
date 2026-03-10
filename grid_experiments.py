import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from dataset import load_data
from linear_training import linear_classification_no_SMOTE, linear_classification_SMOTE
from random_forest_training import random_forest_no_SMOTE, random_forest_with_SMOTE
from xgboost_training import xgboost_no_SMOTE, xgboost_with_SMOTE

def apply_thresholding(X, threshold=0.5):
    """
    Apply thresholding to the time series data.
    Converts values below threshold to 0, above to 1.
    """
    return np.where(X < threshold, 0, 1)

def run_experiment_grid():
    """
    Run experiments with different combinations of:
    1. SMOTE (True/False)
    2. Thresholding (True/False)
    3. Models (Linear, Random Forest, XGBoost)
    """
    # Load the original data
    print("Loading data...")
    (X_train, X_train_reverse, Y_train, ids_train), (X_test, X_test_reverse, Y_test, ids_test), metadata = load_data()
    
    # Flatten the 3D time series data to 2D for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Convert one-hot encoded labels back to class labels
    y_train_labels = np.argmax(Y_train, axis=1)
    y_test_labels = np.argmax(Y_test, axis=1)
    
    results = []
    
    # Experiment grid
    for use_smote in [False, True]:
        for use_threshold in [False, True]:
            print(f"\nRunning experiment: SMOTE={use_smote}, Threshold={use_threshold}")
            
            # Prepare data based on experiment settings
            X_train_processed = X_train_flat.copy()
            X_test_processed = X_test_flat.copy()
            
            # Apply thresholding if needed
            if use_threshold:
                X_train_processed = apply_thresholding(X_train_processed)
                X_test_processed = apply_thresholding(X_test_processed)
            
            # Apply SMOTE if needed
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_labels)
                Y_train_resampled = np.eye(Y_train.shape[1])[y_train_resampled]
            else:
                X_train_resampled, y_train_resampled = X_train_processed, y_train_labels
                Y_train_resampled = Y_train
            
            # Train and evaluate each model
            models = {
                "Linear": (linear_classification_no_SMOTE, linear_classification_SMOTE),
                "RandomForest": (random_forest_no_SMOTE, random_forest_with_SMOTE),
                "XGBoost": (xgboost_no_SMOTE, xgboost_with_SMOTE)
            }
            
            for model_name, (no_smote_func, smote_func) in models.items():
                print(f"  Training {model_name}...")
                
                try:
                    # Select the appropriate function based on SMOTE setting
                    if use_smote:
                        model_results = smote_func(X_train_resampled, Y_train_resampled, X_test_processed, Y_test)
                    else:
                        model_results = no_smote_func(X_train_resampled, Y_train_resampled, X_test_processed, Y_test)
                    
                    # Add experiment metadata to results
                    result_entry = {
                        "model": model_name,
                        "smote": use_smote,
                        "threshold": use_threshold,
                        "precision": model_results.get("precision", 0),
                        "recall": model_results.get("recall", 0),
                        "f1_score": model_results.get("f1_score", 0),
                        "roc_auc": model_results.get("roc_auc", 0)
                    }
                    
                    results.append(result_entry)
                    print(f"    {model_name} - F1: {result_entry['f1_score']:.4f}")
                    
                except Exception as e:
                    print(f"    Error training {model_name}: {str(e)}")
    
    return results

def save_results(results, filename="experiment_results.csv"):
    """
    Save experiment results to a CSV file.
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Print summary
    print("\nExperiment Results Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    print("Starting experiment grid...")
    results = run_experiment_grid()
    save_results(results)
    print("Experiment grid completed!")
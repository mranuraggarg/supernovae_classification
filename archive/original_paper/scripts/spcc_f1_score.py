import os
import json
import numpy as np
import torch
import pandas as pd
import xgboost as xgb
from joblib import load
from dataset import load_saved_data

# Define SPCC F1 score
def f1_score(y_pred, y_true):
    '''
    Calculating SPCC F1-score
    = (1/ (TP + FN))x (TP^2 / (TP + 3 x FP))

    TP: True Positive
    FN: False Negative
    FP: False Positive
    '''

    negative = 0.0
    positive = 1.0

    TP = np.sum(np.logical_and(y_pred == positive, y_true == positive))
    TN = np.sum(np.logical_and(y_pred == negative, y_true == negative))
    FP = np.sum(np.logical_and(y_pred == positive, y_true == negative))
    FN = np.sum(np.logical_and(y_pred == negative, y_true == positive))
    
    spcc_f1_score = TP**2 / ((TP + FN) * (TP + (3 * FP)))
    return spcc_f1_score

if __name__ == "__main__":
    # Load the saved data
    data = load_saved_data(format="npz")
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = data

    # Preprocess the data for the model
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten for consistency
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Loading scalars
    scalar = load("models/xgboost_optimized/xgboost_scaler.pkl")
    X_test_scaled = scalar.transform(X_test_flat)

    # Load the final model xgb_model.pkl
    model = xgb.Booster()
    model.load_model("models/xgboost_optimized/xgboost_optimized.json")

    # Convert test data to DMatrix
    dtest = xgb.DMatrix(X_test_scaled)

    # Predict probabilities
    Y_pred_probs = model.predict(dtest)

    # Convert Probabilities to Class Labels
    Y_pred = (Y_pred_probs >= 0.5).astype(int)

    print(f1_score(Y_pred, Y_test_labels))
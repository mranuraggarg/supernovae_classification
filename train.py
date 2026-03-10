from linear_training import linear_classification_no_SMOTE, linear_classification_SMOTE
from random_forest_training import random_forest_no_SMOTE, random_forest_with_SMOTE
from xgboost_training import xgboost_no_SMOTE, xgboost_with_SMOTE
# train_xgboost_optimized
from dataset import load_saved_data
import pandas as pd


if __name__ == "__main__":
    # ✅ Load dataset
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = load_saved_data(format="npz")

    print("🚀 Starting Linear Model Training...")
    result_linear_no_SMOTE = linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_no_SMOTE)
    print("✅ Linear model (without SMOTE) saved in models/linear/")

    result_linear_SMOTE = linear_classification_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_SMOTE)
    print("✅ Linear model (with SMOTE) saved in models/linear/")

    print("\n🚀 Starting Random Forest Training...")
    rf_results_without_smote = random_forest_no_SMOTE(X_train, Y_train, X_test, Y_test)
    rf_results_with_smote = random_forest_with_SMOTE(X_train, Y_train, X_test, Y_test)
    print("\n📊 Random Forest Results:")
    print(pd.DataFrame([rf_results_without_smote, rf_results_with_smote]).to_markdown(index=False))

    print("\n🚀 Starting XGBoost Training...")
    xgb_results_without_smote = xgboost_no_SMOTE(X_train, Y_train, X_test, Y_test)
    xgb_results_with_smote = xgboost_with_SMOTE(X_train, Y_train, X_test, Y_test)
    print("\n📊 XGBoost Results:")
    print(pd.DataFrame([xgb_results_without_smote, xgb_results_with_smote]).to_markdown(index=False))
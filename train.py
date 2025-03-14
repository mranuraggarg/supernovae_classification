from linear_training import linear_classification_no_SMOTE, linear_classification_SMOTE
from random_forest_training import random_forest_classification
from xgboost_training import train_xgboost_no_smote, train_xgboost_with_smote, train_xgboost_optimized
from dataset import load_saved_data
import pandas as pd


if __name__ == "__main__":
# ✅ Load dataset
    X_train, Y_train, X_test, Y_test, _, _, _, _, _ = load_saved_data(format="npz")

    # ✅ Train both Linear models (with and without SMOTE)
    result_linear_no_SMOTE = linear_classification_no_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_no_SMOTE)
    print("Linear classification model without SMOTE is saved in folder models/linear")

    result_linear_SMOTE = linear_classification_SMOTE(X_train, Y_train, X_test, Y_test)
    print(result_linear_SMOTE)
    print("Linear classification model without SMOTE is saved in folder models/linear")

    # ✅ Train both RF models (with and without SMOTE)
    rf_results_without_smote = random_forest_classification(X_train, Y_train, X_test, Y_test, use_smote=False)
    rf_results_with_smote = random_forest_classification(X_train, Y_train, X_test, Y_test, use_smote=True)

    print("\n📊 **Final Results**")
    print(pd.DataFrame([rf_results_without_smote, rf_results_with_smote]).to_markdown(index=False))

    # ✅ Train all xgboost models
    train_xgboost_no_smote(X_train, Y_train, X_test, Y_test)
    train_xgboost_with_smote(X_train, Y_train, X_test, Y_test)
    train_xgboost_optimized(X_train, Y_train, X_test, Y_test)
    # print(xgboost_results)
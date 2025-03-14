from dataset import load_saved_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
from sklearn.utils import resample

# Load the saved data
data = load_saved_data(format="npz")
X_train, Y_train, X_test, Y_test, _, _, _, _, _ = data

# Preprocess the data for the model
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten for consistency
Y_test_labels = np.argmax(Y_test, axis=1)

# Loading scalars
scalar = joblib.load("models/xgboost_optimized/xgboost_scaler.pkl")
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

# ✅ Perform Bootstrap Resampling
n_iterations = 1000  # Number of bootstrap samples
precision_scores, recall_scores, f1_scores, roc_auc_scores, pr_auc_scores = [], [], [], [], []

for _ in range(n_iterations):
    # ✅ Resample Test Set (with replacement)
    indices = resample(range(len(Y_test_labels)), replace=True)
    y_true_sample = Y_test_labels[indices]
    y_pred_probs_sample = model.predict(dtest)[indices]
    y_pred_sample = (y_pred_probs_sample >= 0.5).astype(int)

    # ✅ Compute Metrics
    precision_scores.append(precision_score(y_true_sample, y_pred_sample, average="weighted"))
    recall_scores.append(recall_score(y_true_sample, y_pred_sample, average="weighted"))
    f1_scores.append(f1_score(y_true_sample, y_pred_sample, average="weighted"))
    roc_auc_scores.append(roc_auc_score(y_true_sample, y_pred_probs_sample))
    precision, recall, thresholds = precision_recall_curve(y_true_sample, y_pred_sample)
    pr_auc_scores.append(auc(recall, precision))

# ✅ Compute Mean & Confidence Intervals (95% CI)
def confidence_interval(scores):
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return np.mean(scores), lower, upper

# Compute the confusion matrix and classification report
final_confusion_matrix = confusion_matrix(Y_test_labels, Y_pred)
final_classification_report = classification_report(Y_test_labels, Y_pred, output_dict=True)
final_classification_report_df = pd.DataFrame(final_classification_report)
final_classification_report_df.columns = ["Non Type Ia", "Type Ia", "accuracy", "macro avg", "weighted avg"]

# Plot the confusion matrix
df_cm = pd.DataFrame(final_confusion_matrix, index=["Non Type Ia", "Type Ia"], columns=["Non Type Ia", "Type Ia"])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Model Confusion Matrix")
plt.savefig("plots/final_model_confusion_matrix.png", dpi=300, bbox_inches="tight")

# ✅ Adjust display settings
pd.set_option("display.max_colwidth", None)  # Ensure full text visibility
pd.set_option("display.float_format", "{:.4f}".format)  # Format floats for uniformity

# Print the classification report
print("\nFinal Model Classification Report:")
print(final_classification_report_df.to_markdown(index=True, tablefmt="grid"))  # `grid` improves readability

# Calculate the ROC Curve
fpr, tpr, _ = roc_curve(Y_test_labels, Y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Breast Cancer Classification')
plt.legend()
plt.savefig("plots/final_model_roc_curve.png", dpi=300, bbox_inches="tight")

# Calculate the Precision-Recall Curve
precision, recall, _ = precision_recall_curve(Y_test_labels, Y_pred_probs)
pr_auc = auc(recall, precision)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
# plt.plot([1, 0], [0, 1], 'k--', label='No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig("plots/final_model_precision_recall_curve.png", dpi=300, bbox_inches="tight")

precision_mean, precision_lower, precision_upper = confidence_interval(precision_scores)
recall_mean, recall_lower, recall_upper = confidence_interval(recall_scores)
f1_mean, f1_lower, f1_upper = confidence_interval(f1_scores)
roc_auc_mean, roc_auc_lower, roc_auc_upper = confidence_interval(roc_auc_scores)
pr_auc_mean, pr_auc_lower, pr_auc_upper = confidence_interval(pr_auc_scores)

# Compute the final model metrics
precision = precision_score(Y_test_labels, Y_pred, average="weighted")
recall = recall_score(Y_test_labels, Y_pred, average="weighted")
f1 = f1_score(Y_test_labels, Y_pred, average="weighted")
roc_auc = roc_auc_score(Y_test_labels, Y_pred_probs)
precision_list, recall_list, thresholds = precision_recall_curve(Y_test_labels, Y_pred_probs)
pr_auc = auc(recall_list, precision_list)

# ✅ Display Results
metrics_results = {
    " ": ["Final Model on Test dataset", "Bootstrap Mean", "Bootstrap Lower", "Bootstrap Upper"],
    "Precision": [precision, precision_mean, precision_lower, precision_upper],
    "Recall": [recall, recall_mean, recall_lower, recall_upper],
    "F1-Score": [f1, f1_mean, f1_lower, f1_upper],
    "ROC-AUC": [roc_auc, roc_auc_mean, roc_auc_lower, roc_auc_upper],
    "PR-AUC": [pr_auc, pr_auc_mean, pr_auc_lower, pr_auc_upper]
}

metrics_df = pd.DataFrame(metrics_results, index=["Final Model", "Bootstrap Mean", "Bootstrap Lower", "Bootstrap Upper"])

# ✅ Adjust display settings
# pd.set_option("display.max_colwidth", None)  # Ensure full text visibility
# pd.set_option("display.float_format", "{:.4f}".format)  # Format floats for uniformity

# ✅ Print Final Table with improved formatting
print("\n🔹 **Final Model Metrics with Confidence Intervals** (95% CI) (Over 1000 cycles)**")
print(metrics_df.to_markdown(index=False, tablefmt="grid"))  # `grid` improves readability

# ✅ Bar Plot of Final Model Metrics
metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
means = [precision_mean, recall_mean, f1_mean, roc_auc_mean, pr_auc_mean]
lower_error = [precision_mean - precision_lower, recall_mean - recall_lower, f1_mean - f1_lower, roc_auc_mean - roc_auc_lower, pr_auc_mean - pr_auc_lower]
upper_error = [precision_upper - precision_mean, recall_upper - recall_mean, f1_upper - f1_mean, roc_auc_upper - roc_auc_mean, pr_auc_upper - pr_auc_mean]
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
plt.bar(metrics, means, yerr=[lower_error, upper_error], capsize=5, color="skyblue")
for val in [precision_mean, recall_mean, f1_mean, roc_auc_mean, pr_auc_mean]:
    plt.text(metrics.index(metrics[means.index(val)]), val - 0.05, f"{val:.4f}", ha="center", va="bottom")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Final Model Metrics with 95% Confidence Intervals")
plt.savefig("plots/final_model_metrics.png", dpi=300, bbox_inches="tight")

# Compute Confusion Matrix
cm = confusion_matrix(Y_test_labels, Y_pred)
fp = cm[0, 1]  # False Positives
fn = cm[1, 0]  # False Negatives

# Plot Misclassification Breakdown
plt.figure(figsize=(8, 5))
sns.barplot(x=["False Positives", "False Negatives"], y=[fp, fn], 
            hue=["False Positives", "False Negatives"], 
            palette="Reds_r", legend=False)
plt.ylabel("Count")
plt.title("Misclassification Analysis")
plt.savefig("plots/misclassification_analysis.png", dpi=300, bbox_inches="tight")

# Compute ROC Curve
fpr, tpr, _ = roc_curve(Y_test_labels, Y_pred)

# Plot Cumulative Gain Curve
plt.figure(figsize=(8, 6))
sns.lineplot(x=fpr, y=tpr, label="Model", color="blue")
sns.lineplot(x=[0, 1], y=[0, 1], label="Random", linestyle="dashed", color="black")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Cumulative Gain)")
plt.title("Cumulative Gain Curve")
plt.legend()
plt.savefig("plots/cumulative_gain_curve.png", dpi=300, bbox_inches="tight")

# Plot Prediction Probability Distribution
plt.figure(figsize=(8, 5))
sns.histplot(Y_pred[Y_test_labels == 0], label="Non-Ia (0)", color="blue", kde=True, stat="density", bins=30)
sns.histplot(Y_pred[Y_test_labels == 1], label="Ia (1)", color="red", kde=True, stat="density", bins=30)
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.title("Prediction Probability Distribution")
plt.legend()
plt.savefig("plots/prediction_probability_distribution.png", dpi=300, bbox_inches="tight")
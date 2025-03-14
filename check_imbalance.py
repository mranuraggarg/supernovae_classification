from dataset import load_saved_data
import matplotlib.pyplot as plt
from matplotlib import rc
rc("text", usetex=True)
import numpy as np
import pandas as pd
import seaborn as sns

data = load_saved_data(format="npz")
X_train, Y_train, X_test, Y_test, _, _, _, _, _ = data

# ✅ Check for class imbalance
df_train = pd.DataFrame(Y_train)
df_test = pd.DataFrame(Y_test)
df_train["class"] = df_train[0].apply(lambda x: "Type Ia" if x == 1 else "Non Type Ia")
df_test["class"] = df_test[0].apply(lambda x: "Type Ia" if x == 1 else "Non Type Ia")
df_train.drop([0, 1], axis=1, inplace=True)
df_test.drop([0, 1], axis=1, inplace=True)

# Calculate the total dataset size
minority_class = df_train[df_train["class"] == "Type Ia"].shape[0] + df_test[df_test["class"] == "Type Ia"].shape[0]
majority_class = df_train[df_train["class"] == "Non Type Ia"].shape[0] + df_test[df_test["class"] == "Non Type Ia"].shape[0]
total_dataset_size = majority_class + minority_class
print(f"Total Non Type Ia (majority class): {majority_class}")
print(f"\nTotal Type Ia (minority class): {minority_class}")
print(f"\nTotal Dataset Size: {total_dataset_size}")

# Calculate Imbalance Ratio
IR_ratio = majority_class / minority_class
print(f"\nImbalance Ratio: {IR_ratio:.4f}")
print(f"\nNumber of Non Type Ia in Training Data: {df_train[df_train['class'] == 'Non Type Ia'].shape[0]}")
print(f"\nNumber of Type Ia in Training Data: {df_train[df_train['class'] == 'Type Ia'].shape[0]}")


# Plot the class distribution
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.barplot(x=df_train["class"].value_counts().index, y=[majority_class, minority_class])
plt.text(0, majority_class - 600, f"Non Type Ia: {majority_class}", fontsize=12, color="white", ha="center")
plt.text(1, minority_class - 600, f"Type Ia: {minority_class}", fontsize=12, color="white", ha="center")
plt.text(1, majority_class, f"Imbalance Ration: {IR_ratio:.4f}", fontsize=12, color="red", ha="center")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.title(r"\textbf{Class Distribution in Training Data}", fontsize=14, color="black", ha="center")
plt.savefig("plots/class_distribution.png", dpi=300, bbox_inches="tight")
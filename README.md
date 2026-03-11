# Supernovae Type Ia Classification — Phase 2 Tier‑1

This branch represents **Phase‑2 Tier‑1 of the project**, where the focus shifts from repairing the original pipeline (Phase‑1) to **owning the full preprocessing pipeline and building an interpretable compact feature baseline**.  

Phase‑2 introduces:

• A reproducible **feature engineering pipeline derived directly from light‑curve statistics**  
• A **compact 16‑feature baseline model** validated through ablation and importance analysis  
• **Interpretability analysis (SHAP + permutation importance)** to understand the astrophysical signals driving classification  
• A structured experimental workflow designed to support future studies such as feature ablation, early classification, and uncertainty estimation.

A complete machine learning pipeline for the photometric classification of Type Ia Supernovae.

----------

**Supernovae Type Ia Classification**
A machine-learning pipeline for  **Type Ia Supernovae classification**, leveraging  **XGBoost, Random Forest, and Linear Models**. This repository is inspired by  [Adam Moss’s Supernovae Dataset](https://github.com/adammoss/supernovae)  and extends it with optimized training and evaluation techniques.

**📌 Project Overview**
The workflow in Phase‑2 Tier‑1 consists of four stages:

1. **Raw Data Preprocessing**  
   Construction of light curves and extraction of physically interpretable summary features.

2. **Feature Engineering**  
   Generation of brightness, color, variability, and temporal features derived from multi‑band photometric observations.

3. **Model Training**  
   Gradient boosted tree models (XGBoost) are trained on progressively reduced feature sets.

4. **Interpretability and Validation**  
   Feature importance, SHAP analysis, and ablation studies are used to identify a compact and scientifically meaningful feature representation.

The repository allows users to either  **run pre-trained models**  or  **train models from scratch**, depending on their requirements.

----------

**🛠️ Installation and Setup**

This project uses  **Conda**  for environment management to ensure reproducibility.

**1️⃣ Create and Activate Conda Environment**

```bash
conda env create -f environment.yml
conda activate ds_new  # Ensure the environment name matches  
```

**2️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/supernovae-classification.git
cd supernovae-classification
```
**🚀 Usage Guide**

  

This repository supports three different execution modes based on user needs.

  

**1️⃣ Running Pre-Trained Models**

  

If you only want to evaluate pre-trained models and skip training, run:
```bash
python main.py
```

• **What happens?**

•  Loads the existing models from  models/{model_name}

•  Uses the preprocessed dataset  supernovae_dataset.npz

•  Evaluates all six models on the held-out test split and presents results in a structured format

  

✅ **No training or preprocessing is performed** in this mode.

----------

**2️⃣ Training All Models**

  

If you wish to  **train all six models**  from scratch using preprocessed data:

```bash
python train.py
```
• **What happens?**

•  Uses the dataset stored in  supernovae_dataset.npz

•  Trains six models sequentially:

• **Linear Models (with and without SMOTE)**

• **XGBoost (with and without SMOTE)**

• **Random Forest (with and without SMOTE)**

•  Saves trained models in  models/{model_name}

•  Generates a results summary using validation-tuned models and final test-set evaluation

  

⚠️  **Note:**  Training all models is computationally expensive and was originally performed using:

• **Colab A100 GPU** for **XGBoost training**

• **Apple M1 GPU** for **Linear Model training**

  

🚀  **Recommended:**  Run this step on a machine with a  **powerful GPU**.

----------

**3️⃣ Full Pipeline Execution**

  

To  **start from raw data**  and execute the entire workflow:

```bash
tar -xvf SIMGEN_PUBLIC_DES.tar.gz && python preprocess.py && python train.py && python main.py
```
• **What happens?**

1. **Extracts raw supernovae dataset** (from SIMGEN_PUBLIC_DES.tar.gz)

2. **Runs preprocessing** (preprocess.py) to generate supernovae_dataset.npz

3. **Trains all six models** (train.py) and saves them in models/{model_name}

4.  **Evaluates trained models**  (main.py) and presents a final summary

  

📊  **Final Output:**  A structured  **Pandas DataFrame**  comparing all trained models.

----------

**📂 Repository Structure**

```
supernovae_classification/
├── README.md
├── environment.yml
├── astro-ml.yml
├── main.py                    # Evaluate saved models
├── train.py                   # Train all model variants
├── preprocess.py              # Build supernovae_dataset.npz from raw inputs
├── evaluate.py                # Shared evaluation helpers
├── dataset.py                 # Dataset loading utilities
├── linear_training.py
├── random_forest_training.py
├── xgboost_training.py
├── grid_experiments.py        # Grid-search experiment runner
├── analysing_final_model.py   # Final model analysis and plots
├── check_imbalance.py
├── spcc_f1_score.py
├── data/                      # Input CSV shards
├── models/
│   ├── original_paper/
│   └── phase1_repair/
├── results/
│   ├── original_paper/
│   └── phase1_repair/
├── plots/
│   ├── original_paper/
│   └── phase1_repair/
├── notebooks/                 # Exploratory notebooks
├── scripts/
├── notes/
└── supernovae_dataset.npz     # Cached preprocessed dataset
```

For the full file listing, see `repo_tree.txt`.
----------

----------

📘 **Phase‑2 Tier‑1 Experimental Design**

Phase‑2 introduces a structured feature‑selection workflow.

Feature sets evaluated:

| Configuration | Feature Count | Purpose |
|---------------|--------------|--------|
| Full Baseline | 31 | Original engineered feature pool |
| Working Set | 30 | Removal of clearly redundant features |
| Compact Baseline | 16 | Final interpretable feature set |

The compact baseline preserves model performance while improving interpretability and reducing redundancy.

Final compact baseline performance:

| Metric | Value |
|------|------|
| F1 | **0.8442** |
| ROC‑AUC | **0.9766** |
| PR‑AUC | **0.9278** |

These results demonstrate that a carefully engineered feature set can match the predictive power of larger models while remaining scientifically interpretable.

----------

📊 **Interpretability Analysis**

Phase‑2 Tier‑1 emphasizes understanding *why* the classifier works.

Two complementary analyses are used:

• **Permutation importance** – measures performance degradation when features are shuffled.

• **SHAP values** – quantify each feature's contribution to individual predictions.

The SHAP summary plot for the compact baseline can be found at:

```
results/phase2_tier1/phase2_tier1_compact_baseline_plots/phase2_tier1_compact_baseline_shap_summary.png
```

Key physical signals learned by the model:

• Brightness scale across photometric bands  
• Color gradients between filters  
• Temporal ordering of peak emission  
• Light‑curve variability structure

These correspond directly to known observational characteristics of Type Ia supernovae.

----------

**📈 Models Implemented**
|**Model Name**|**Data Balancing**  |**Algorithm Used** |
|--|--| --|
| Linear (Without SMOTE) |No  | MLP |
|Linear (With SMOTE)|Yes| MLP |
| XGBoost (Without SMOTE) | No | Gradient Boosting Trees |
|XGBoost (With SMOTE)|Yes| Gradient Boosting Trees |
|Random Forest (Without SMOTE)  | No |Random Forest |
|Random Forest (With SMOTE)|Yes| Random Forest|

  

We train  **six different models**  to compare their performance.  
The results below reflect the repaired Phase 1 evaluation pipeline, where hyperparameter tuning is performed on a validation split and final metrics are reported on an untouched test set.

----------

**📊 Performance Summary**
|**Model**  |**Precision** | **Recall**| **F1-Score** | **ROC-AUC** | **PR-AUC** |
| -- | -- | -- | -- | -- | -- |
| Linear (Without SMOTE) | 0.7446 | 0.7253 | 0.7334 | 0.7223 | 0.8873 |
| Linear (With SMOTE) | 0.7472 | 0.6753 | 0.6971 | 0.7250 | 0.8896 |
| Random Forest (Without SMOTE) | 0.9010 | 0.9031 | 0.8992 | 0.9645 | 0.9893 |
| Random Forest (With SMOTE) | 0.9239 | 0.9209 | 0.9220 | 0.9679 | 0.9905 |
| XGBoost (Without SMOTE) | 0.9226 | 0.9238 | 0.9229 | 0.9750 | **0.9926** |
| XGBoost (With SMOTE) | **0.9269** | **0.9252** | **0.9258** | **0.9732** | 0.9922 |

🛠  **Final selection:** **XGBoost (Without SMOTE)** remains the preferred model when prioritizing ranking performance (**highest ROC-AUC and PR-AUC**), while **XGBoost (With SMOTE)** provides the best thresholded classification metrics (**precision, recall, and F1-score**).

----------

----------

**📊 Paper vs Phase‑1 Repaired Results**

The table below compares the originally reported paper results with the repaired Phase‑1 evaluation pipeline implemented in this repository.

| Model | Metric | Paper | Repaired | Δ (Repaired − Paper) | Interpretation |
|------|------|------|------|------|------|
| Linear (No SMOTE) | Precision | 0.714 | 0.7446 | +0.0306 | Material change – baseline evaluation corrected |
|  | Recall | 0.714 | 0.7253 | +0.0113 | Small increase |
|  | F1 | 0.714 | 0.7334 | +0.0194 | Moderate increase |
|  | ROC‑AUC | 0.000 | 0.7223 | +0.7223 | Evaluation bug fixed |
| Linear (SMOTE) | Precision | 0.685 | 0.7472 | +0.0622 | Material change – corrected baseline |
|  | Recall | 0.685 | 0.6753 | −0.0097 | Small change |
|  | F1 | 0.685 | 0.6971 | +0.0121 | Small increase |
|  | ROC‑AUC | 0.000 | 0.7250 | +0.7250 | Evaluation bug fixed |
| Random Forest (No SMOTE) | Precision | 0.902 | 0.9010 | −0.0010 | Negligible change |
|  | Recall | 0.905 | 0.9031 | −0.0019 | Negligible change |
|  | F1 | 0.902 | 0.8992 | −0.0028 | Negligible change |
|  | ROC‑AUC | 0.965 | 0.9645 | −0.0005 | Negligible change |
| Random Forest (SMOTE) | Precision | 0.927 | 0.9239 | −0.0031 | Small change |
|  | Recall | 0.922 | 0.9209 | −0.0011 | Negligible change |
|  | F1 | 0.924 | 0.9220 | −0.0020 | Small change |
|  | ROC‑AUC | 0.970 | 0.9679 | −0.0021 | Small change |
| XGBoost (No SMOTE) | Precision | 0.926 | 0.9226 | −0.0034 | Small change |
|  | Recall | 0.927 | 0.9238 | −0.0032 | Small change |
|  | F1 | 0.927 | 0.9229 | −0.0041 | Small change |
|  | ROC‑AUC | 0.975 | 0.9750 | ~0 | No meaningful change |
| XGBoost (SMOTE) | Precision | 0.931 | 0.9269 | −0.0041 | Small change |
|  | Recall | 0.927 | 0.9252 | −0.0018 | Negligible change |
|  | F1 | 0.928 | 0.9258 | −0.0022 | Small change |
|  | ROC‑AUC | 0.972 | 0.9732 | +0.0012 | Negligible change |

**Summary:**

• Ensemble models (XGBoost and Random Forest) remain stable after methodological repair.

• The linear baseline shows the largest change because the original evaluation pipeline produced invalid ROC‑AUC values.

• The repaired pipeline confirms that XGBoost remains the strongest model while preserving the original scientific conclusions.

**📚 Acknowledgments**

•  **Data Source:**  The dataset and preprocessing pipeline are borrowed from  [Adam Moss’s Supernovae Dataset](https://github.com/adammoss/supernovae).

•  **Inspiration:**  Inspired by existing works in Type Ia Supernovae classification.

----------

**📜 License**

  

This project is licensed under the  **MIT License**. Feel free to modify and use it as per your needs.

----------

**💡 Future Work**
•  **Hyperparameter Optimization**: Further fine-tuning XGBoost with Bayesian Optimization.

•  **Ensemble Learning**: Combining multiple models to improve accuracy.

•  **Deep Learning**: Exploring Transformer-based approaches for classification.

• **Feature Ablation Study (Phase‑2 Tier‑2)** – systematic removal of compact features to quantify astrophysical signal strength.

• **Early Classification** – evaluate how early in the light curve a reliable Ia classification can be made.

• **Uncertainty / Abstention Framework** – allow the classifier to abstain on ambiguous events.

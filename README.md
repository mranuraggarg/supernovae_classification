# Supernovae Type Ia Classification

A complete machine learning pipeline for the photometric classification of Type Ia Supernovae.

----------

**Supernovae Type Ia Classification**
A machine-learning pipeline for  **Type Ia Supernovae classification**, leveraging  **XGBoost, Random Forest, and Linear Models**. This repository is inspired by  [Adam Moss’s Supernovae Dataset](https://github.com/adammoss/supernovae)  and extends it with optimized training and evaluation techniques.

**📌 Project Overview**
This project provides an end-to-end pipeline for classifying Type Ia Supernovae using multiple machine-learning models. The workflow consists of three key stages:
1.  **Preprocessing**  - Converts raw data into a structured format.
2.  **Model Training**  - Trains six models with validation-based hyperparameter optimization.
3.  **Evaluation**  - Evaluates the final selected models on an untouched test dataset and presents final results.

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

----------

**🔬 Phase 2 Tier‑1 SPCC Compact Baseline**

This branch (`phase2-tier1-spcc-preprocessing`) adds a native owned SPCC Tier‑1 preprocessing and feature-selection workflow on top of the repaired Phase‑1 model line.

The official Tier‑1 working baseline is:

`phase2_tier1_compact_baseline`

It uses the following 16 features:

- `z_peak_flux`
- `r_mean_flux`
- `peak_color_g_minus_r`
- `i_peak_flux`
- `peak_color_r_minus_i`
- `peak_color_i_minus_z`
- `g_mean_flux`
- `r_peak_flux`
- `z_std_flux`
- `i_amplitude`
- `i_std_flux`
- `time_span`
- `z_time_of_peak`
- `i_time_of_peak`
- `r_time_of_peak`
- `r_std_flux`

**Official Tier‑1 Summary**

| Name | Feature Count | F1 | ROC-AUC | PR-AUC | Δ F1 vs Full | Δ ROC-AUC vs Full | Δ PR-AUC vs Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31-feature full baseline | 31 | 0.836717 | 0.976449 | 0.928071 | +0.000000 | +0.000000 | +0.000000 |
| 30-feature working set | 30 | 0.840529 | 0.976398 | 0.928204 | +0.003812 | -0.000050 | +0.000133 |
| 16-feature compact baseline | 16 | 0.844230 | 0.976588 | 0.927761 | +0.007513 | +0.000140 | -0.000310 |

Interpretation:

• The compact baseline improves F1 relative to both larger baselines.

• ROC-AUC is marginally better than the 31-feature full baseline.

• PR-AUC is effectively preserved, with only a very small drop.

----------

**🔁 Reproducing Phase 2 Tier‑1 Results**

The commands below reproduce the Tier‑1 SPCC results in this branch using the working local environment:

```bash
cd /Volumes/AstroSSD/share/github/supernovae_classification
/opt/miniconda3/envs/astro-ml/bin/python ...
```

**0. Input data layout**

Expected raw input:

- `data/spcc/raw/DES_SN*.DAT`
- `data/spcc/raw/DES_UNBLIND+HOSTZ.KEY`
- `data/spcc/raw/DES_UNBLINDnoHOSTZ.KEY`

**1. Build native Tier‑1 feature artifacts**

```bash
/opt/miniconda3/envs/astro-ml/bin/python -m feature_pipeline.extraction.spcc_features --input-glob 'data/spcc/raw/DES_SN*.DAT'
```

Outputs:

- `data/processed/spcc_features_tier1.csv`
- `data/processed/spcc_features_tier1.npz`
- `results/phase2_tier1/spcc_tier1_metadata.json`

**2. Validate features and generate diagnostic plots**

```bash
MPLCONFIGDIR=/Volumes/AstroSSD/share/github/supernovae_classification/__pycache__/mpl \
PYTHONPYCACHEPREFIX=/Volumes/AstroSSD/share/github/supernovae_classification/__pycache__ \
/opt/miniconda3/envs/astro-ml/bin/python -m feature_pipeline.validation.checks
```

Outputs:

- `results/phase2_tier1/spcc_feature_validation_report.json`
- `results/phase2_tier1/spcc_candidate_feature_summary.json`
- `plots/phase2_tier1/`

**3. Run the compact-feature benchmark comparison**

```bash
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_benchmarks.py
```

Output:

- `results/phase2_tier1/phase2_tier1_benchmark_results.json`

**4. Run XGBoost nonlinear importance on the fixed split**

```bash
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_xgb_importance.py
```

Output:

- `results/phase2_tier1/phase2_tier1_xgb_importance.json`

**5. Build the first manifest and run review-feature ablation**

```bash
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_build_feature_manifest.py
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_review_ablation.py
```

Outputs:

- `results/phase2_tier1/phase2_tier1_feature_manifest.json`
- `results/phase2_tier1/phase2_tier1_review_ablation.json`

**6. Tighten the manifest and rerun the compact baseline**

```bash
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_tighten_manifest.py
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_compact_rerun.py
```

Outputs:

- `results/phase2_tier1/phase2_tier1_feature_manifest_tightened.json`
- `results/phase2_tier1/phase2_tier1_compact_rerun.json`

**7. Finalize the official compact baseline**

```bash
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_finalize_compact_baseline.py
```

Outputs:

- compact manifest:
  - `results/phase2_tier1/phase2_tier1_compact_baseline_manifest.json`
- compact dataset:
  - `data/processed/phase2_tier1_compact_baseline.csv`
  - `data/processed/phase2_tier1_compact_baseline.npz`
- trained compact XGBoost model:
  - `models/phase2_tier1/compact_baseline/phase2_tier1_compact_baseline_xgb.json`
- compact metrics:
  - `results/phase2_tier1/phase2_tier1_compact_baseline_metrics.json`
- compact importance:
  - `results/phase2_tier1/phase2_tier1_compact_baseline_importance.json`
- official comparison table:
  - `results/phase2_tier1/phase2_tier1_compact_baseline_comparison.csv`
  - `results/phase2_tier1/phase2_tier1_compact_baseline_comparison.md`
- robustness pass:
  - `results/phase2_tier1/phase2_tier1_compact_baseline_robustness.json`

**8. Run interpretability analysis without retraining**

```bash
MPLCONFIGDIR=/Volumes/AstroSSD/share/github/supernovae_classification/__pycache__/mpl \
PYTHONPYCACHEPREFIX=/Volumes/AstroSSD/share/github/supernovae_classification/__pycache__ \
/opt/miniconda3/envs/astro-ml/bin/python phase2_tier1_interpretability.py
```

Outputs:

- `results/phase2_tier1/phase2_tier1_compact_baseline_interpretability.json`
- `results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.csv`
- `results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.md`
- `results/phase2_tier1/phase2_tier1_compact_baseline_plots/`

**9. Expected compact-baseline metrics**

From the finalized compact baseline:

- F1: `0.8442299254`
- ROC-AUC: `0.9765883838`
- PR-AUC: `0.9277608810`

**10. Robustness summary**

The compact baseline was checked across 3 seeds:

- F1 mean: `0.842998`
- F1 std: `0.001783`
- ROC-AUC mean: `0.976723`
- ROC-AUC std: `0.000219`
- PR-AUC mean: `0.928472`
- PR-AUC std: `0.000985`

----------

**📚 Acknowledgments**

•  **Data Source / Historical Reference:**  This repository builds on the public supernovae resources and earlier repository context from  [Adam Moss’s Supernovae Dataset](https://github.com/adammoss/supernovae).

•  **Phase 2 Tier‑1 Preprocessing:**  The native SPCC Tier‑1 preprocessing, feature extraction, feature selection, compact baseline, and interpretability workflow on this branch are owned implementations in this repository.

•  **Inspiration:**  Inspired by existing works in Type Ia Supernovae classification.

----------

**📜 License**

  

This project is licensed under the  **MIT License**. Feel free to modify and use it as per your needs.

----------

**💡 Future Work**
•  **Hyperparameter Optimization**: Further fine-tuning XGBoost with Bayesian Optimization.

•  **Ensemble Learning**: Combining multiple models to improve accuracy.

•  **Deep Learning**: Exploring Transformer-based approaches for classification.

----------

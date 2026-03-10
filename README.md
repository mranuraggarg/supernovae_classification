# Supernovae Type Ia Classification

[![DOI](https://zenodo.org/badge/948372312.svg)](https://doi.org/10.5281/zenodo.15074652)

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
supernovae-classification/
│── data/                        # Raw and processed data
│   ├── SIMGEN_PUBLIC_DES.tar.gz  # Raw dataset (original)
│   ├── supernovae_dataset.npz    # Preprocessed dataset
│
│── models/                      # Trained models directory
│   ├── linear/
│   ├── xgboost/
│   ├── random_forest/
│
│── scripts/                      # Python scripts for different tasks
│   ├── preprocess.py             # Preprocesses raw data
│   ├── train.py                  # Trains all models
│   ├── main.py                   # Runs pre-trained models for evaluation
│
│── environment.yml               # Conda environment setup
│── .gitignore                    # Files to ignore in Git
│── README.md                     # Project documentation
```
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

----------

## 📜 Citation

If you use this code or models in your research, please cite the Zenodo record:

**DOI**: [10.5281/zenodo.15074652](https://doi.org/10.5281/zenodo.15074652)

Anurag Garg. (2025). mranuraggarg/supernovae_classification: v1.0 — Type Ia Supernovae Classification with Emphasis on F1-Score and PR-AUC (v1.0). Zenodo. https://doi.org/10.5281/zenodo.15074653

**Stroke Classification Using Explainable Artificial Intelligence (XAI)**

## Project Overview

This project investigates the application of Explainable Artificial Intelligence (XAI) to stroke classification using machine learning. Two classification models, Random Forest and Support Vector Machine (SVM), are evaluated and interpreted using SHapley Additive exPlanations (SHAP).

The project examines predictive performance, global feature importance and individual prediction explanations, with particular attention to class imbalance and the limitations of interpreting machine learning outputs in healthcare.


## Dataset

**Stroke Prediction Dataset — Kaggle**

Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The dataset contains:
- 5,110 observations
- 12 original attributes, including the target variable
- 249 recorded stroke cases (4.87%)
- 201 missing BMI values

The target variable, `stroke`, indicates recorded stroke status (0 = no recorded stroke; 1 = recorded stroke).

## Models and Explanation Methods

### Random Forest — Tree SHAP

A Random Forest classifier with 100 decision trees was trained using random oversampling of the minority class.

Tree SHAP was applied to explain:
- Individual predictions using local force plots
- Global feature contributions using summary plots
- Additive reconstruction of predicted probabilities

### Support Vector Machine — Kernel SHAP

An SVM classifier with an RBF kernel, balanced class weights and feature scaling was used as the alternative black-box model.

Kernel SHAP was applied using:
- 30 background training observations
- 40 held-out testing observations
- `nsamples=100`
- Random seed 2021

## Model Performance

Both models were evaluated on the same held-out testing dataset containing 1,022 observations.

| Metric | Random Forest | SVM |
|---|---:|---:|
| Accuracy | 94.03% | 75.34% |
| Balanced accuracy | 51.33% | 69.96% |
| Stroke-class precision | 13.33% | 12.03% |
| Stroke-class recall | 4.00% | 64.00% |
| F1-score | 0.0615 | 0.2025 |
| ROC AUC | 0.8270 | 0.8016 |
| Average precision | 0.1454 | 0.1622 |

The Random Forest achieved higher overall accuracy but detected only 2 of the 50 recorded stroke cases in the testing dataset. The SVM identified 32 positive cases, although it also generated more false positives.

## Key SHAP Findings

- **Random Forest:** Age, average glucose level and BMI were the most influential global features.
- **SVM:** Age, BMI and hypertension were the leading global features.
- **Local explanations:** Both models incorrectly classified the selected recorded stroke case (test observation 32) as negative.
- **Interpretability:** SHAP helped explain model behaviour but did not establish causal relationships or clinical validity.

## Notebook

The main implementation is available in:

`XAIASSIGNMENT1.ipynb`

The notebook contains exploratory data analysis, preprocessing, model training, evaluation, Tree SHAP and Kernel SHAP explanations, and visualization exports.

The implementation was developed in Google Colab using Python and the supplied XAI teaching repository. Reproduction requires the dataset and supporting utility files referenced by the notebook.

## Technologies

Python, Google Colab, NumPy, pandas, scikit-learn, imbalanced-learn, SHAP, Matplotlib and Seaborn.

## Limitations

This project uses a retrospective educational dataset. The results are affected by class imbalance, dataset representativeness and limitations in model validation. The models were not externally or prospectively validated, and probability calibration was not formally assessed.

**The predictions must not be interpreted as clinically validated estimates of future stroke risk or used independently for medical decision-making.**

## Acknowledgements

Dataset: Stroke Prediction Dataset, Kaggle.

The implementation was developed using teaching materials and a supplied Python repository for the Explainable Artificial Intelligence course at Universiti Malaya.

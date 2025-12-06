# Credit-Card-Fraud
Build a machine learning model to detect fraudulent credit card transactions with extremely imbalanced data (fraud = 0.17%)
# Data Preparation & Feature Engineering

1. Dataset

284,807 transactions
31 features: PCA-derived V1–V28, Amount, Time, Class

2. New engineered features

Hour → extracted from Time
Day → derived from Hour
Amount_scaled → standardized for ML stability

3. Observations

Fraud is extremely rare (492 cases → 0.17%)
Fraud mostly occurs late night (Hour ≈ 23, 4–6 AM)
Fraud amounts show higher spread than legit transactions
Strong fraud predictors: V14, V12, V10, V17, Amount

# Exploratory Data Analysis (Key Insights)

1. Fraud transactions tend to be:

Higher in amount
More variable (wide IQR)
Occurring at suspicious hours
Linked strongly to PCA components V14, V12, V10

2. Correlation Insights

Top features correlated with fraud:
Strong negative: V14, V12, V10
Moderate: V17, V11
Slight positive: Amount

# Modeling Strategy

Tested 2 approaches:

A) Logistic Regression + class_weight balanced

Very high recall on fraud
Many false positives
Good baseline


C) XGBoost (Final Model)

This became your best model.
XGBoost handles imbalance + complex patterns extremely well.

# Final Model Metrics (optimal threshold):

Precision (Fraud)	0.97
Recall (Fraud)	0.76
F1-score	0.85
ROC-AUC	0.9796

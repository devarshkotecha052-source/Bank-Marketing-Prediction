# Bank Marketing Prediction

## Project Overview

This project uses machine learning models to predict whether a customer is likely to subscribe to a bank term deposit.

The application provides two prediction modes:

- Single Customer Prediction
- Batch CSV Campaign Prediction

The Streamlit application allows users to enter customer information or upload a CSV file, select a machine learning model for batch prediction, view prediction probabilities, identify high-potential customers, and download prediction results.

## Dataset

The project uses the Bank Marketing dataset.

The dataset contains customer information, campaign information, previous campaign history, and the target variable representing term-deposit subscription.

## Machine Learning Models

The project includes the following 14 trained models:

1. Logistic Regression
2. Decision Tree
3. Bagging Classifier
4. Random Forest
5. Extra Trees
6. AdaBoost
7. Gradient Boosting
8. XGBoost
9. Tuned Logistic Regression
10. Tuned AdaBoost
11. Tuned Gradient Boosting
12. Tuned XGBoost
13. Stacking Ensemble
14. Soft Voting Ensemble

## Final Model

The project selected the Soft Voting Ensemble as the final predictive model.

Final test performance:

- Accuracy: 89.13%
- Precision: 55.36%
- Recall: 36.58%
- F1-Score: 44.05%

## Application Features

### Single Customer Mode

Users can enter:

- Age
- Balance
- Day of contact
- Current campaign contacts
- Previous contact information
- Job
- Marital status
- Education
- Credit default
- Housing loan
- Personal loan
- Contact type
- Month
- Previous campaign outcome

The application generates predictions and probabilities from all 14 models.

### Batch CSV Mode

Users can:

1. Select one machine learning model.
2. Upload a customer CSV file.
3. Generate predictions for all uploaded customers.
4. View predicted Yes/No outcomes.
5. View Yes and No probabilities.
6. Identify customers with higher predicted subscription probability.
7. Download the prediction results.

## Project Structure

```text
Bank-Marketing-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── bank_marketing_adaboost_model.pkl
    ├── bank_marketing_adaboost_tuned_model.pkl
    ├── bank_marketing_bagging_model.pkl
    ├── bank_marketing_dt_model.pkl
    ├── bank_marketing_extra_trees_model.pkl
    ├── bank_marketing_feature_columns.pkl
    ├── bank_marketing_gradient_boosting_model.pkl
    ├── bank_marketing_gradient_boosting_tuned_model.pkl
    ├── bank_marketing_lr_model.pkl
    ├── bank_marketing_lr_tuned_model.pkl
    ├── bank_marketing_rf_model.pkl
    ├── bank_marketing_stacking_model.pkl
    ├── bank_marketing_voting_model.pkl
    ├── bank_marketing_xgboost_model.pkl
    └── bank_marketing_xgboost_tuned_model.pkl
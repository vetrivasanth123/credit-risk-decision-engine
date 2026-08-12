# Credit Risk Decision Engine

## Overview

An end-to-end credit risk decision system that predicts probability of default,
calibrates the predicted probabilities, converts them into lending decisions,
and quantifies portfolio-level expected loss.

## Business Problem

Build a system that converts customer information into:

- Probability of Default (PD)
- Risk Band
- Lending Decision
- Expected Loss

## Dataset

This project uses the **Home Credit Default Risk** dataset.

The raw dataset is not included in this repository due to
dataset size and repository management considerations.

Dataset source:

https://www.kaggle.com/c/home-credit-default-risk/data

### Dataset used in this project

- Training customers: 172,205
- Held-out test customers: 43,052
- Raw/engineered input features: 106
- Model features after preprocessing: 226

The repository contains the final notebook, model artifacts,
preprocessing pipeline, calibration model, decision configuration,
and project report required to understand and reproduce the
deployed pipeline after obtaining the original dataset.

## Methodology

1. Data preparation
2. Feature engineering
3. Preprocessing
4. Baseline modelling
5. LightGBM modelling
6. Global SHAP explainability
7. Probability calibration
8. OOF validation
9. Decision engine
10. Risk segmentation
11. Expected-loss analysis
12. Deployment validation
13. Stress testing
14. Governance audit

## Model

LightGBM classifier with probability calibration.

## Performance

| Metric | Result |
|---|---:|
| Test ROC-AUC | 0.7395 |
| Test PR-AUC | 0.2264 |
| Brier Score | 0.0691 |
| OOF ROC-AUC | 0.7340 |
| OOF Std | 0.0018 |
| Train ROC-AUC | 0.8571 |
| Train-Test Gap | 0.1176 |

## Decision Engine

| Decision | Rule |
|---|---|
| APPROVE | PD < 0.10 |
| REVIEW | 0.10 <= PD < 0.18 |
| REJECT | PD >= 0.18 |

## Risk Management

The system assigns customers to four risk bands:

- Low Risk
- Medium Risk
- High Risk
- Very High Risk

Expected loss is estimated using:

PD × Exposure × LGD

## Business Impact

High and Very High Risk customers represent approximately 28.3% of customers
but approximately 56.0% of expected portfolio loss.

## Deployment

The project packages:

- preprocessing pipeline
- LightGBM model
- calibration model
- feature metadata
- decision configuration
- model metadata

The packaged artifacts were verified through an end-to-end
deployment pipeline and reproduced the original model predictions
without numerical differences.  

The deployment pipeline was tested using:

- single-customer prediction
- batch prediction
- missing values
- extreme numerical values
- unseen categorical values
- deterministic repeated predictions
- invalid feature-count protection

## Limitations

The main limitation is the train-test ROC-AUC gap of approximately 0.118,
indicating remaining overfitting.

Future improvements include:

stronger regularization
hyperparameter comparison
cost-sensitive threshold optimization
customer-level/local SHAP explanations
production drift monitoring

## Project Report

See:

`reports/credit_risk_decision_engine_project_report.pdf`

## Notebook

See:

`notebooks/credit_risk_final.ipynb`

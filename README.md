# Credit Risk Decision Engine

## Overview

An end-to-end credit risk decision system that predicts probability of default, calibrates predicted probabilities, converts them into lending decisions, segments customers by risk, and quantifies portfolio-level expected loss.

## Business Problem

Build a system that converts customer information into:

- Probability of Default (PD)
- Risk Band
- Lending Decision
- Expected Loss
- Model explainability

## Dataset

This project uses the **Home Credit Default Risk** dataset.

The raw dataset is not included in this repository due to dataset size and repository management considerations.

Dataset source:

https://www.kaggle.com/c/home-credit-default-risk/data

### Dataset used in this project

| Item | Value |
|---|---:|
| Training customers | 172,205 |
| Held-out test customers | 43,052 |
| Original input columns | 99 |
| Engineered input features | 7 |
| Raw/engineered input features | 106 |
| Model features after preprocessing | 226 |
| Observed test default rate | 8.11% |

The repository contains the final notebook, model artifacts, preprocessing pipeline, calibration model, decision configuration, and project report required to understand and reproduce the deployed pipeline after obtaining the original dataset.

## Methodology

1. Data preparation
2. Exploratory analysis
3. Feature engineering
4. Train/test split
5. Preprocessing
6. Baseline modelling
7. LightGBM modelling
8. Global SHAP explainability
9. Probability calibration
10. OOF validation
11. Decision engine
12. Risk segmentation
13. Expected-loss analysis
14. Deployment validation
15. Stress testing
16. Governance audit

## Feature Engineering

The project creates seven engineered features:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_GOODS_RATIO`
- `INCOME_PER_FAMILY_MEMBER`
- `CREDIT_PER_FAMILY_MEMBER`
- `EMPLOYMENT_INCOME_RATIO`
- `EXT_SOURCE_MEAN`

The resulting pipeline contains 106 raw/engineered inputs and 226 processed model features.

## Model

**LightGBM classifier with probability calibration.**

A Logistic Regression baseline achieved:

- ROC-AUC: 0.7260
- PR-AUC: 0.2035

The final LightGBM model achieved:

- ROC-AUC: 0.7395
- PR-AUC: 0.2264

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

Five-fold OOF ROC-AUC ranged from 0.7318 to 0.7364.

## Explainability

Global SHAP explainability was completed using `shap.TreeExplainer`.

- SHAP sample: 5,000 held-out customers
- Model features analysed: 226
- Global mean absolute SHAP importance generated
- SHAP summary plot generated
- Risk-direction analysis generated

The top transformed model features by mean absolute SHAP value include `Column_91`, `Column_24`, `Column_87`, `Column_25`, and `Column_4`.

The SHAP analysis is post-hoc explainability and does not change the final model or official evaluation results. The transformed feature names are not assigned business meanings without a verified mapping to original inputs.

## Probability Calibration

The LightGBM raw probability is passed to a one-feature calibration model.

- Calibrated Brier Score: 0.0691
- Maximum absolute calibration gap: 0.0095
- Mean absolute calibration gap: 0.0048

## Decision Engine

| Decision | Rule | Customer Share |
|---|---|---:|
| APPROVE | PD < 0.10 | 71.71% |
| REVIEW | 0.10 <= PD < 0.18 | 17.22% |
| REJECT | PD >= 0.18 | 11.07% |

## Risk Management

The system assigns customers to four risk bands:

- Low Risk
- Medium Risk
- High Risk
- Very High Risk

Expected loss is estimated using:

**Expected Loss = PD × Exposure × LGD**

Exposure variable: `AMT_CREDIT`

LGD assumption: `0.45`

## Business Impact

| Metric | Result |
|---|---:|
| Customers evaluated | 43,052 |
| Portfolio exposure | 25,974,998,023.50 |
| Portfolio average PD | 0.0835 |
| Expected loss | 905,828,407.72 |
| Expected loss rate | 3.4873% |
| High + Very High customer share | 28.29% |
| High + Very High expected-loss share | 55.96% |
| Very High expected-loss share | 28.98% |

**Key finding:** High and Very High Risk customers represent approximately 28.3% of customers but approximately 56.0% of expected portfolio loss.

## Deployment

The project packages:

- preprocessing pipeline
- LightGBM model
- calibration model
- feature metadata
- decision configuration
- model metadata

The packaged artifacts were reloaded and verified to reproduce the original model probabilities and calibrated PDs without numerical differences.

The inference flow is:

`106 raw/engineered inputs -> preprocessing -> 226 model features -> LightGBM raw PD -> calibration -> calibrated PD -> decision/risk band`

## Deployment Validation

The deployment pipeline was tested using:

- single-customer prediction
- 100-customer batch prediction
- missing values
- extreme numerical values
- unseen categorical values
- deterministic repeated predictions
- decision-threshold boundary checks
- invalid feature-count protection

All tests passed.

## Governance

Governance covered:

- model performance
- OOF/generalization
- decision-engine configuration
- business impact
- deployment artifacts
- stress testing

**Governance completeness: 100%**

**Governance grade: EXCELLENT**

## Limitations

- Train-test ROC-AUC gap of approximately 0.118 indicates remaining overfitting.
- Test ROC-AUC of 0.7395 leaves room for stronger discrimination.
- Decision thresholds are validated business rules, not claimed to be globally cost-optimal.
- Global SHAP analysis is completed, but a full customer-level/local explanation interface is not implemented.
- Live production drift, calibration and performance monitoring is not implemented.

## Future Improvements

- Stronger regularization and validation
- Alternative model and hyperparameter comparison
- Additional domain-specific feature engineering
- Cost-sensitive threshold optimization
- Customer-level/local SHAP explanations
- Production drift, calibration and performance monitoring

## Project Report

See:

`reports/credit_risk_decision_engine_project_report.pdf`

## Notebook

See:

`notebooks/credit_risk_final.ipynb`

## Final Project Status

Project 1 is complete and frozen. The official model, preprocessing, calibration, thresholds and evaluation results are not changed by the documentation update.

# Credit Risk Decision Engine

An end-to-end credit risk decision system built with LightGBM and post-hoc Isotonic probability calibration. The system converts raw customer financial attributes into calibrated Probability of Default (PD), operational lending decisions, risk segmentation bands, portfolio expected-loss (EL) estimates, SHAP explainability, and a stress-tested deployment pipeline.

## Project Structure

credit-risk-decision-engine/
├── artifacts/
│   ├── calibration_model.pkl
│   ├── decision_config.pkl
│   ├── feature_names.pkl
│   ├── lgbm_model.pkl
│   ├── model_metadata.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── credit_risk_final.ipynb
├── reports/
│   └── credit_risk_decision_engine_project_report.pdf
├── .gitignore
├── README.md
└── requirements.txt

## Business Problem & Objectives

In commercial lending, raw classification probabilities dictate underwriting decisions, capital reserve allocation, and risk-based pricing. The core objectives of this system are to:

* Predict continuous, un-skewed Probability of Default (PD).
* Systematically reduce tree model overfitting and control generalization variance.
* Calibrate model output probabilities to align with empirical portfolio default rates.
* Map calibrated PD to automated **APPROVE / REVIEW / REJECT** lending workflows.
* Construct four operational risk bands with strictly monotonic default rates.
* Estimate portfolio expected financial loss ($\text{EL} = \text{PD} \times \text{EAD} \times \text{LGD}$).
* Provide global model auditability using SHAP TreeExplainer.
* Package and stress-test an end-to-end deployment pipeline.



## Dataset Overview

This project utilizes the [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) benchmark dataset. The dataset is split 80/20 via stratified sampling to preserve class ratios.

| Parameter | Value / Count |

| **Training Customers** | 172,205 |
| **Held-Out Test Customers** | 43,052 |
| **Original Input Columns** | 99 *(after removing 1 constant column)* |
| **Engineered Domain Features** | 7 |
| **Total Raw / Engineered Inputs** | 106 |
| **Initial Encoded Model Features** | 226 |
| **Final Pruned Model Features** | 186 |
| **Observed Test Base Rate** | 8.12% |


## Feature Engineering

To evaluate household capacity, debt burden, and external score metrics, 7 high-impact domain ratios were engineered:

* `CREDIT_INCOME_RATIO`: Total credit liability relative to total annual income.
* `ANNUITY_INCOME_RATIO`: Annual loan payment relative to total annual income.
* `CREDIT_GOODS_RATIO`: Total credit financed relative to the purchased goods price.
* `INCOME_PER_FAMILY_MEMBER`: Household income distributed per family member.
* `CREDIT_PER_FAMILY_MEMBER`: Household debt liability distributed per family member.
* `EMPLOYMENT_AGE_RATIO`: Employment duration relative to applicant age.
* `EXT_SOURCE_MEAN`: Consolidated mean across external credit score metrics (`EXT_SOURCE_1`, `2`, `3`, `4`).



## Model Development & Experimental Progression

Model development progressed across 5 structured experimental iterations targeting generalization gap reduction, class imbalance optimization, feature pruning, and probability calibration.

| Experiment Stage | Train ROC-AUC | Test ROC-AUC | Test PR-AUC | Generalization Gap | Key Strategy / Outcome |

| **Baseline LightGBM** | 0.8571 | 0.7395 | 0.2264 | 0.1176 | Initial baseline; severe structural overfitting |
| **Exp 1: Regularization** | 0.7946 | 0.7439 | 0.2314 | 0.0507 | Reduced generalization gap by 57% |
| **Exp 2: Hyperparameter Tuning** | 0.8022 | 0.7445 | 0.2321 | 0.0578 | Tuned depth, max leaves, and subsampling |
| **Exp 3: Class Imbalance Optimization** | 0.7905 | 0.7443 | 0.2322 | 0.0462 | Applied `scale_pos_weight=3.0` |
| **Exp 4: Feature Pruning** | 0.7968 | 0.7452 | 0.2329 | 0.0516 | Pruned noisy features from 226 down to 186 |
| **Exp 5: Isotonic Calibration** | **0.7968** | **0.7452** | **0.2270** | **0.0516** | Champion Model; Brier score corrected to 0.0688 |

### Cross-Validation & Stability
* **5-Fold Stratified OOF Mean ROC-AUC**: `0.7403 ± 0.0018`
* **Train-Test Generalization Gap**: Reduced by **>55%** (from `0.1176` baseline down to `0.0516`).



## Probability Calibration

Because tree classifiers trained with weighted instances (`scale_pos_weight=3.0`) produce distorted probability estimations, post-hoc **Isotonic Regression** (`calibration_model.pkl`) was fitted on out-of-fold predictions.

| Metric | Uncalibrated Model | Calibrated Model | Status / Impact |

| **Brier Score** | 0.0839 | **0.0688** | Improved probability accuracy (lower is better) |
| **Average Test PD** | 18.82% | **8.12%** | Aligned with true portfolio base rate (8.12%) |
| **Test ROC-AUC** | 0.7452 | **0.7452** | Preserved ranking performance |
| **Test PR-AUC** | 0.2329 | **0.2270** | Preserved precision-recall curve shape |



## Decision Engine & Operational Workflows

Calibrated probabilities pass through a deterministic rule engine that assigns operational lending actions based on fixed risk boundaries.

| Decision Action | Rule Threshold | Customer Share (%) | Applicant Count | Observed Default Rate |

| **APPROVE** | $\text{PD} < 0.10$ | 71.71% | 30,872 | ~4.5% |
| **REVIEW** | $0.10 \le \text{PD} < 0.18$ | 17.22% | 7,414 | ~12.9% |
| **REJECT** | $\text{PD} \ge 0.18$ | 11.07% | 4,766 | ~27.8% |



## Risk Segmentation

Applicants are grouped into 4 risk tiers to enable risk-based pricing, capital reserve provisioning, and credit limit structuring.

| Risk Tier | PD Band | Customer Share (%) | Avg Calibrated PD | Observed Default Rate |

| **Low Risk (Tier 1)** | $\text{PD} < 0.05$ | 44.10% | 3.13% | 2.85% |
| **Medium Risk (Tier 2)** | $0.05 \le \text{PD} < 0.10$ | 27.61% | 7.11% | 6.95% |
| **High Risk (Tier 3)** | $0.10 \le \text{PD} < 0.18$ | 17.22% | 13.40% | 12.93% |
| **Very High Risk (Tier 4)** | $\text{PD} \ge 0.18$ | 11.07% | 24.41% | 27.85% |



## Expected Loss & Portfolio Impact

Expected financial loss (EL) is evaluated across held-out test accounts using standard Basel risk metrics:

$$\text{Expected Loss (EL)} = \text{PD} \times \text{Exposure at Default (EAD)} \times \text{Loss Given Default (LGD)}$$

* **Exposure (EAD)**: Total credit amount (`AMT_CREDIT`).
* **LGD Assumption**: Fixed at 45% ($0.45$).

| Portfolio Financial Metric | Evaluated Value |

| **Evaluated Test Customers** | 43,052 |
| **Total Evaluated Portfolio Exposure (EAD)** | $25,974,998,023.50 |
| **Average Calibrated Portfolio PD** | 8.12% |
| **Total Expected Portfolio Loss (EL)** | **$948,882,732.45** |
| **Portfolio Expected Loss Rate** | 3.65% |
| **High + Very High Risk Share** | 28.29% of applicants |
| **High + Very High Risk Expected Loss Contribution** | **~56.0%** ($531.37M) |

> **Key Risk Insight**: Applicants in High and Very High risk tiers constitute **28.29%** of applicants but generate **~56.0%** of total expected financial loss. Automated rejections and manual reviews targeted at these tiers prevent over half of potential portfolio losses.



## Global Explainability (SHAP)

Global feature importance was computed using `shap.TreeExplainer` on a representative sample of 5,000 held-out test customers.

1. **`EXT_SOURCE_MEAN`**: Consolidated external score is the strongest negative driver of default (higher scores substantially lower PD).
2. **`CREDIT_INCOME_RATIO` & `ANNUITY_INCOME_RATIO`**: Higher debt service burdens correlate directly with increased default likelihood.
3. **`DAYS_BIRTH` & `DAYS_EMPLOYMENT`**: Younger applicants and shorter employment histories drive higher risk scores.
4. **`CREDIT_GOODS_RATIO`**: Over-leveraged financing relative to underlying asset value increases default probability.



## Deployment & Production Pipeline

The production pipeline is modularized and serialized in the `artifacts/` folder:

* **`preprocessor.pkl`**: Combined `ColumnTransformer` (median imputation, standard scaling, one-hot encoding).
* **`feature_names.pkl`**: Serialized list of the 186 pruned feature names.
* **`lgbm_model.pkl`**: Trained Champion LightGBM classifier.
* **`calibration_model.pkl`**: Isotonic probability calibration model.
* **`decision_config.pkl`**: Frozen decision thresholds and operational policy rules.
* **`model_metadata.pkl`**: Version control, performance metrics, and build lineage metadata.

### Data Flow Execution Sequence

Raw 106 Inputs 
  └─► Preprocessor (226 Encoded Features)
        └─► Feature Selection (186 Pruned Features)
              └─► LightGBM Classifier (Raw PD)
                    └─► Isotonic Calibrator (Calibrated PD)
                          └─► Decision Engine (Lending Action & Risk Tier)


### Stress Testing Validation
The automated deployment suite verified **8/8 test scenarios (100% Pass Rate)**:
* Single-Customer Streaming Payload Inference
* 100-Customer Batch Dataframe Streaming
* Missing / Null Value Handling & Imputation
* Extreme Numerical Outlier Resilience
* Unseen Categorical String Handling (`handle_unknown="ignore"`)
* Deterministic Prediction Consistency Across Runs
* Boundary Threshold Evaluation (PD at 0.05, 0.10, and 0.18)
* Malformed Input Payload Rejection



## Governance & Audit Summary

| Audit Dimension | Evaluation Status | Score |

| **Performance & Overfitting Audit** | Complete | Passed |
| **Cross-Validation Stability Audit** | Complete | Passed |
| **Decision Monotonicity Audit** | Complete | Passed |
| **Business Loss Concentration Audit** | Complete | Passed |
| **Deployment Artifact Serialization Audit**| Complete | Passed |
| **Stress Testing Suite Audit** | Complete | Passed |
| **Overall Governance Score** | **100.0%** | **EXCELLENT** |



## Project Artifacts & Documentation

* **Full Execution Notebook**: [`notebooks/credit_risk_final.ipynb`](notebooks/credit_risk_final.ipynb)
* **Comprehensive PDF Report**: [`reports/credit_risk_decision_engine_project_report.pdf`](reports/credit_risk_decision_engine_project_report.pdf)



## Limitations & Future Roadmap

* **Residual Tree Overfitting**: A train-test gap of `0.0516` remains due to gradient boosting capacity on tabular financial data.
* **Static Risk Boundaries**: Decision thresholds ($\text{PD} < 0.10$, $\text{PD} \ge 0.18$) are static business rules rather than dynamically adjusted parameters tied to macroeconomic interest rate shifts.
* **Future Work**: Explore neural tabular architectures (TabNet / FT-Transformer), implement live Population Stability Index (PSI) drift monitoring, and integrate real-time REST API endpoints.

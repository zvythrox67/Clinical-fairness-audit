# Cross-Domain Fairness Auditing: Credit and Health Risk Models

A research prototype that trains ML risk models on **credit** and **health** data, audits them for demographic disparity, and tests whether standard fairness mitigation techniques generalize across domains.

> **⚠️ Research prototype only.** This is not financial advice, not a medical device, not a diagnostic tool, and not intended for real-world credit or clinical decisions. All analysis runs on publicly available historical data with no live deployment.

---

## Overview

Machine learning models used in lending and healthcare can silently encode historical bias. This project:

1. Trains an XGBoost classifier on **1.35 million completed LendingClub loans** (2007–2018)
2. Trains a second XGBoost classifier on **253,680 CDC BRFSS diabetes survey responses**
3. Audits both models for fairness across protected groups using Fairlearn
4. Applies **AIF360 Reweighing** as a pre-processing bias mitigation technique in both domains
5. Reports the **cross-domain finding**: Reweighing is unreliable — it backfires in credit and fails entirely in health

The unexpected results — and the analysis of why they happen — are the primary contribution.

---

## Headline Results

| Domain | Model | DP Diff | EO Diff |
|---|---|---|---|
| Credit (Loan) | Baseline | 0.027 | 0.063 |
| Credit (Loan) | Reweighed | **0.239** | **0.232** |
| Health (Diabetes) | Baseline | 0.488 | 0.466 |
| Health (Diabetes) | Reweighed | **0.488** | **0.466** |

**Three findings:**
1. **Baseline disparity is 18× larger in health than credit** (0.488 vs 0.027)
2. **Reweighing backfired in credit** (DP Diff 0.027 → 0.239)
3. **Reweighing failed entirely in health** (NaN weights, no change)

---

## Domain 1: Credit Risk (Loan Default)

### Setup

- **Data:** LendingClub accepted loans, 2007–2018 (1.35M completed loans)
- **Target:** `1` = Charged Off, `0` = Fully Paid
- **Features:** 16 origination-time features including loan amount, interest rate, DTI, FICO score, revolving balance, and derived ratios
- **Protected attribute:** `income_group` (income quartiles Q1–Q4)
- **Model:** XGBoost, 100 trees, max depth 5, `scale_pos_weight` set to class ratio (~4.0)

### Baseline Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.64 |
| ROC-AUC | 0.71 |
| Recall on defaults | 0.68 |

### Fairness Audit (Baseline)

| Group | Selection Rate | FPR | TPR |
|---|---|---|---|
| Q1 (lowest income) | 0.032 | 0.018 | 0.080 |
| Q2 | 0.024 | 0.014 | 0.062 |
| Q3 | 0.015 | 0.008 | 0.045 |
| Q4 (highest income) | 0.005 | 0.002 | 0.017 |

**DP Diff: 0.027 · EO Diff: 0.063**

Q1 applicants are flagged as default risks at ~7× the rate of Q4 applicants.

### After Reweighing

| Metric | Baseline | Reweighed |
|---|---|---|
| DP Diff | 0.027 | **0.239** |
| EO Diff | 0.063 | **0.232** |

Reweighing **widened** the disparity instead of narrowing it.

### Hypothesis

The credit result stems from an interaction between two rebalancing mechanisms:

- `scale_pos_weight` already pushes the model toward predicting defaults aggressively to compensate for the 80/20 class imbalance
- AIF360 Reweighing further upweights minority-outcome instances per protected group, assuming balanced favorable/unfavorable outcomes

On imbalanced credit data, the two mechanisms **compound rather than correct each other**, producing a wider selection-rate gap.

### Visualizations

<img width="805" height="284" alt="image" src="https://github.com/user-attachments/assets/aa6ed1a8-77a6-4c49-bc1a-24e1a7acbc95" />

*Left: fairness metrics before and after Reweighing. Right: selection rate by income group — note the wider spread after mitigation.*

<img width="472" height="350" alt="image" src="https://github.com/user-attachments/assets/498e9b2b-7cc8-4193-b1eb-24c281eb373b" />

*Top features: `int_rate`, `loan_to_income`, `revol_bal`. Interest rate is a proxy for LendingClub's own risk assessment, which is itself influenced by historical lending patterns.*

<img width="388" height="335" alt="image" src="https://github.com/user-attachments/assets/b2cc4ad5-d6f8-4c71-ba4f-d9e02c2d1f01" />

*With class weighting, the model trades precision for recall — it catches 68% of defaults but flags many non-defaults as risky.*

---

## Domain 2: Health Risk (Diabetes Prediction)

To test whether the credit findings generalize to healthcare, the entire pipeline was replicated on the **CDC Diabetes Health Indicators dataset** (BRFSS 2015).

### Setup

- **Data:** CDC BRFSS 2015 diabetes health indicators (253,680 survey responses)
- **Target:** `1` = prediabetes or diabetes, `0` = no diabetes (14% positive)
- **Features:** 21 health, lifestyle, and demographic indicators (BMI, blood pressure, cholesterol, smoking, physical activity, general health, etc.)
- **Protected attribute:** `age_group` (binary: Young = ages 18–39, Older = ages 40+)
- **License:** CC0 1.0 Public Domain (CDC)
- **Model:** Same XGBoost configuration

### Baseline Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.722 |
| ROC-AUC | 0.827 |
| Recall on diabetes | 0.79 |
| Precision on diabetes | 0.31 |

ROC-AUC of 0.827 is at the level of published BRFSS benchmarks.

### Fairness Audit (Baseline)

| Group | Selection Rate | FPR | TPR |
|---|---|---|---|
| Young | 0.044 | 0.032 | 0.380 |
| Older | 0.532 | 0.455 | 0.846 |

**DP Diff: 0.488 · EO Diff: 0.466**

Older adults are flagged as diabetes-risk at **12× the rate** of younger adults.

### Why Age Disparity is Different from Income Disparity

Unlike the credit model's income disparity, the age disparity in the diabetes model is **not evidence of algorithmic bias**.

Age is the **second-strongest predictor** of diabetes in the model (feature importance 489, after BMI at 560), consistent with medical literature — diabetes prevalence rises sharply with age.

A model with high demographic parity across age groups would be a **less accurate medical model**. This illustrates the fundamental tension between fairness and accuracy in clinical ML: when the protected attribute is causally tied to the outcome, demographic parity conflicts with clinical validity.

### After Reweighing

| Metric | Baseline | Reweighed |
|---|---|---|
| DP Diff | 0.488 | 0.488 |
| EO Diff | 0.466 | 0.466 |

Reweighing produced **no change**. AIF360's internal contingency table contained sparse cells (Young + diabetes), causing division-by-zero and NaN weights. XGBoost silently ignored the NaN weights, leaving the model identical to baseline.

### Visualizations

<img width="396" height="353" alt="image" src="https://github.com/user-attachments/assets/02761520-a801-4874-815e-0a22abb892db" />


*The diabetes model catches 79% of diabetes cases with a high false-positive rate — the same recall-over-precision trade-off as the credit model.*

<img width="472" height="352" alt="image" src="https://github.com/user-attachments/assets/fee16851-2b3a-4e02-8f82-50de0dc24a48" />


*BMI and Age dominate. The model is correctly reflecting biology, not encoding bias.*

---

## Cross-Domain Findings

<img width="803" height="284" alt="image" src="https://github.com/user-attachments/assets/e84f92aa-b9d1-4ad5-8276-0240fc07b440" />



| Domain | Model | DP Diff | EO Diff |
|---|---|---|---|
| Credit (Loan) | Baseline | 0.027 | 0.063 |
| Credit (Loan) | Reweighed | 0.239 | 0.232 |
| Health (Diabetes) | Baseline | 0.488 | 0.466 |
| Health (Diabetes) | Reweighed | 0.488 | 0.466 |

### 1. Disparity is not always bias

Health data shows **18× more measured disparity** than credit data (0.488 vs 0.027). But this doesn't mean the health model is more biased — it means age is *causally linked* to diabetes, while income is only weakly correlated with default risk. Demographic parity is the wrong fairness metric when the protected attribute directly predicts the outcome.

### 2. Pre-processing fairness interventions are domain-dependent

AIF360 Reweighing failed in **both** domains, but for **different reasons**:

- **Credit:** the intervention compounded with class-imbalance correction, widening disparity
- **Health:** sparse contingency cells produced NaN weights, so the intervention did nothing

This inconsistency is the core finding.

### 3. Implication for high-stakes ML

Standard "off-the-shelf" fairness toolkits should not be applied blindly to clinical or financial models. What works in one domain may backfire or silently fail in another. Future work should test:

- Reweighing *without* class-imbalance correction
- Post-processing methods like Fairlearn's `ThresholdOptimizer` that directly equalize selection rates
- In-processing methods like adversarial debiasing
- Domain-specific fairness metrics that account for causal relationships between protected attributes and outcomes

---




# Loan Default Prediction with Fairness Auditing

A research prototype that predicts loan default risk on LendingClub data and audits the resulting model for demographic disparity across income groups.

> **⚠️ Research prototype only.** This is not financial advice, not a lending tool, and not intended for any real-world credit decisions. All simulations run on historical data with no live deployment.

---

## Overview

Machine learning models used in credit scoring can silently encode historical bias. This project:

1. Trains an XGBoost classifier on **1.35 million completed LendingClub loans** (2007–2018)
2. Audits the model for fairness across **income quartiles** using Fairlearn
3. Applies **AIF360 Reweighing** as a pre-processing bias mitigation technique
4. Reports an **unexpected finding**: the mitigation *increased* measured disparity rather than reducing it

The unexpected result — and our analysis of why it happened — is the main contribution.

---

## Results

### Baseline model (XGBoost with class weighting)

| Metric | Value |
|---|---|
| ROC-AUC | 0.71 |
| Accuracy | ~0.64 |
| Recall on defaults | 0.68 |
| Demographic Parity Difference | **0.027** |
| Equalized Odds Difference | **0.063** |

**Selection rate by income quartile (baseline):**

| Group | Selection Rate | FPR | TPR |
|---|---|---|---|
| Q1 (lowest income) | 0.032 | 0.018 | 0.080 |
| Q2 | 0.024 | 0.014 | 0.062 |
| Q3 | 0.015 | 0.008 | 0.045 |
| Q4 (highest income) | 0.005 | 0.002 | 0.017 |

Q1 applicants are flagged as default risks at **~7× the rate** of Q4 applicants — a clear disparity worth investigating.

### After AIF360 Reweighing

| Metric | Baseline | Reweighed |
|---|---|---|
| Demographic Parity Difference | 0.027 | **0.24** |
| Equalized Odds Difference | 0.063 | **0.23** |

**Selection rate by income quartile (reweighed):**

| Group | Selection Rate |
|---|---|
| Q1 | 0.55 |
| Q2 | 0.46 |
| Q3 | 0.40 |
| Q4 | 0.31 |

Reweighing **widened** the disparity instead of narrowing it.

---

## Unexpected Finding

Contrary to expectation, AIF360 Reweighing *increased* demographic disparity in our setting.

**Hypothesis:** The result stems from an interaction between two rebalancing mechanisms:

- `scale_pos_weight` in XGBoost already pushes the model toward predicting defaults aggressively to compensate for the 80/20 class imbalance
- AIF360 Reweighing further upweights minority-outcome instances per protected group, assuming balanced favorable/unfavorable outcomes

On imbalanced credit data, these two mechanisms **compound rather than correct each other**. Reweighing's per-group rebalancing amplifies the class-imbalance correction in a way that diverges across income groups, producing a wider selection-rate gap.

**Implication:** Pre-processing fairness interventions should not be naively combined with class-imbalance correction. Future work should test:

- Reweighing *without* `scale_pos_weight`
- Post-processing methods like Fairlearn's `ThresholdOptimizer`, which directly equalize selection rates
- In-processing methods like adversarial debiasing

This finding is documented as the primary contribution of the project.



## Visualizations

### Fairness Metrics and Selection Rates

(<img width="405" height="281" alt="image" src="https://github.com/user-attachments/assets/32538584-93ee-4c59-9528-301757888a5f" />
)

*Left: Demographic Parity and Equalized Odds differences before and after Reweighing. Right: Selection rate by income group — note the wider spread after mitigation.*

### Feature Importances

(<img width="473" height="353" alt="image" src="https://github.com/user-attachments/assets/32d430b3-5e24-4987-8c99-311e81939243" />
)

*`int_rate`, `loan_to_income`, and `revol_bal` dominate the model. Interest rate is a proxy for LendingClub's own risk assessment, which is itself influenced by historical lending patterns.*

### Confusion Matrix (Baseline)

(<img width="389" height="337" alt="image" src="https://github.com/user-attachments/assets/7b9f2c2c-3205-4235-b96e-9cc875abf807" />
)

*With class weighting, the model trades precision for recall. It catches 68% of defaults but flags many non-defaults as risky — a reasonable choice for a fairness-audit prototype, where having enough positive predictions is necessary for meaningful disparity measurement.*



# Rare-Disease-Diagnostic-Support

---

## Visualizations

### Fairness Metrics and Selection Rates

![Fairness comparison](<img width="405" height="281" alt="image" src="https://github.com/user-attachments/assets/32538584-93ee-4c59-9528-301757888a5f" />
)

*Left: Demographic Parity and Equalized Odds differences before and after Reweighing. Right: Selection rate by income group — note the wider spread after mitigation.*

### Feature Importances

![Feature importances](<img width="473" height="353" alt="image" src="https://github.com/user-attachments/assets/32d430b3-5e24-4987-8c99-311e81939243" />
)

*`int_rate`, `loan_to_income`, and `revol_bal` dominate the model. Interest rate is a proxy for LendingClub's own risk assessment, which is itself influenced by historical lending patterns.*

### Confusion Matrix (Baseline)

![Confusion matrix](<img width="389" height="337" alt="image" src="https://github.com/user-attachments/assets/7b9f2c2c-3205-4235-b96e-9cc875abf807" />
)

*With class weighting, the model trades precision for recall. It catches 68% of defaults but flags many non-defaults as risky — a reasonable choice for a fairness-audit prototype, where having enough positive predictions is necessary for meaningful disparity measurement.*

---

## Setup

### Requirements

```bash
pip install fairlearn xgboost shap aif360

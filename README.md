# Predicting 30-Day Hospital Readmission with XGBoost

## 1. Objective

Hospital readmissions within 30 days of discharge are costly and often preventable. This project builds a machine learning model to predict which patients are at elevated risk of being readmitted within 30 days, with the goal of flagging high-risk patients for additional follow-up care before they leave the hospital.

## 2. Data

The dataset used is the [Synthetic Healthcare Patient Journey Dataset](https://www.kaggle.com/datasets/emirhanakku/synthetic-healthcare-patient-journey-dataset) from Kaggle. As the name states, this is **synthetically generated data**, not real patient records. This has two implications worth stating upfront:

- Absolute performance numbers (AUC, precision, recall) reflect the patterns baked into the data-generating process, not necessarily the true difficulty of predicting readmission in a real clinical setting.
- The methodology — feature engineering, leakage checks, threshold selection, calibration, and fairness analysis — is the transferable, evaluable part of this project.

**Class balance:** 23.5% of patients were readmitted within 30 days; 76.5% were not. This is a moderate imbalance, not extreme, but it shapes how metrics like PR AUC and threshold choice should be interpreted.

**A significant limitation of this dataset:** it contains no patient readmission history (e.g., prior admission count, days since last discharge). In real-world readmission modeling, this kind of utilization history is typically the single strongest predictor available. Its absence here caps how well any model — regardless of algorithm — can perform on this data.

## 3. Methodology

**Preprocessing:**
- Dropped `total_cost_€`, a redundant duplicate of `total_cost_usd`.
- Dropped `patient_id` — see Data Hygiene note below.
- Categorical features (`gender`, `department`, `admission_type`, `discharge_status`) were one-hot encoded rather than label-encoded, since these are nominal categories with no natural order, and label encoding would have implied a false ordinal relationship.

**Data hygiene check — patient_id:** An initial SHAP analysis revealed `patient_id`, a meaningless row identifier, ranked as the **3rd most important feature** in the model. This was a clear signal of the model fitting to noise rather than real signal. After removing it and retraining, ROC AUC and PR AUC changed only marginally, confirming that the model's genuine predictive power was coming from clinically meaningful features rather than a spurious identifier. This is a reassuring result: a large performance drop after removing `patient_id` would have suggested the model was overly reliant on an artifact that would not generalize to new patients.

**Model:** XGBoost classifier (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`).

## 4. Model Performance

Evaluated via 5-fold cross-validation for a more robust estimate than a single train/test split:

| Metric | Score |
|---|---|
| ROC AUC | 0.678 ± 0.021 |
| PR AUC | 0.409 ± 0.024 |

For context, a no-skill classifier would score a PR AUC equal to the base positive rate (0.235). The model's PR AUC of 0.409 represents roughly a **74% improvement over that baseline**, which is a more meaningful way to read the number than in isolation.

## 5. Baseline Comparison

To check whether XGBoost's added complexity was actually earning its keep, it was compared against a simple logistic regression baseline:

| Model | ROC AUC | PR AUC |
|---|---|---|
| XGBoost | 0.678 | 0.409 |
| Logistic Regression | **0.724** | **0.450** |

**The logistic regression baseline outperformed XGBoost on both metrics.** This was an unexpected but instructive result. The most likely explanation is that the relationships in this dataset are largely linear or additive, which logistic regression captures natively, while XGBoost's capacity to model complex non-linear interactions isn't being rewarded here — and may even be mildly overfitting given the dataset's size and feature set. This finding is a useful reminder that a more complex model is not automatically a better one, and that baseline comparisons are a necessary step, not an optional one.

## 6. Threshold Selection

Since ROC AUC and PR AUC evaluate ranking across all thresholds, an explicit operating threshold was chosen to convert probabilities into actionable "flag this patient" decisions. Precision, recall, and total flagged patients were compared across a range of thresholds:

| Threshold | Recall | Precision | Patients Flagged |
|---|---|---|---|
| 0.068 (F1-optimal) | 83.3% | 30.2% | 414 |
| 0.10 | 66.7% | 30.0% | 334 |
| **0.15** | **58.0%** | **32.3%** | **269** |
| 0.20 | 52.7% | 35.7% | 221 |
| 0.30 | 38.7% | 39.7% | 146 |
| 0.40 | 28.7% | 45.7% | 94 |
| 0.50 (default) | 17.3% | 48.1% | 54 |

The default threshold of 0.5 catches only 17.3% of true readmissions despite looking reasonable on paper (higher precision, higher overall accuracy) — a classic case of standard metrics masking poor performance on the metric that actually matters for this use case.

**Threshold 0.15 was selected** as the recommended operating point. It captures 58% of true readmissions while flagging a moderate, targeted 47% of the patient population, under the assumption that in this clinical context, a missed readmission is costlier than an unnecessary follow-up call. This is a judgment call, not a purely mathematical one, and would ideally be refined further with real cost estimates for false positives vs. false negatives.

<img width="1010" height="432" alt="Screenshot 2026-07-16 at 7 46 01 PM" src="https://github.com/user-attachments/assets/e7b5aa61-9220-4ada-8c82-ee5913f3bf85" />


## 7. Interpretability (SHAP)

A SHAP summary plot was used to examine which features drive the model's predictions. After removing `patient_id`, the top features were, in order: `chronic_condition`, `complications`, `age`, `total_cost_usd`, `wait_time_min`, `medication_count`.

These align well with clinical intuition — chronic conditions and complications during the hospital stay are exactly the kind of features that should drive readmission risk. `satisfaction_score` and `discharge_status`, which were flagged early on as possible leakage risks (since they may be recorded at or after discharge), appeared further down the importance ranking rather than dominating it, which is a reassuring sign, though not a complete guarantee against leakage.

A secondary observation: the SHAP plot showed a mild difference in how `gender` affected predictions — female patients showed a slightly stronger positive association with readmission risk than male patients, whose effect clustered closer to neutral. This prompted the fairness check below.

<img width="753" height="843" alt="Screenshot 2026-07-16 at 7 46 17 PM" src="https://github.com/user-attachments/assets/bea3a19d-a7ce-4af1-81dc-a4d88374d7b8" />


## 8. Calibration

A calibration curve revealed that the model's raw predicted probabilities were **well-ranked but poorly calibrated** — the model became increasingly overconfident above a predicted probability of ~0.25, meaning a stated "60% risk" did not correspond to an actual 60% readmission rate among those patients.

Isotonic regression (via `CalibratedClassifierCV`) was applied to correct this. The resulting calibration curve tracked the ideal diagonal closely across the 0.0–0.6 predicted probability range. A deviation remained above ~0.6, but inspection of bin sizes showed this was driven by very small sample counts at the high end (fewer than 10 validation patients scored above 0.6 in total) rather than a genuine calibration failure.

As expected, calibration had minimal effect on ROC AUC (0.631 → 0.640) and PR AUC (0.356 → 0.374), since isotonic regression preserves rank order and these are rank-based metrics. The value of calibration here is in probability *interpretability*, not discrimination — important if predicted probabilities are ever presented to clinicians as literal risk estimates rather than used purely for ranking/thresholding.

<img width="1012" height="412" alt="Screenshot 2026-07-16 at 7 46 51 PM" src="https://github.com/user-attachments/assets/bf8dc87b-4b4f-4af5-9f24-2445150e7a65" />


## 9. Fairness Check

Prompted by the SHAP gender pattern, ROC AUC was computed separately by gender on the validation set:

| Group | ROC AUC | n |
|---|---|---|
| Male | 0.670 | 295 |
| Female | 0.594 | 305 |

This is a meaningful gap (~0.076 AUC) given reasonably balanced sample sizes on both sides — the model discriminates readmission risk noticeably less well for female patients. Since this dataset is synthetic, the root cause can't be fully diagnosed here — it may reflect an artifact of how the data was generated rather than a genuine clinical pattern — but a gap of this size would need investigation before any real-world deployment, and is reported here rather than omitted.

## 10. Limitations & Future Work

- **Synthetic data:** all findings should be understood as a demonstration of methodology, not a claim about real-world readmission risk.
- **No patient history features:** the absence of prior-admission data likely caps model performance well below what would be achievable with real EHR data.
- **Logistic regression outperformed XGBoost:** suggests the relationships in this dataset are largely linear; this would be worth re-testing on real, likely more complex clinical data, and by trying a lighter/less overfit-prone XGBoost configuration.
- **Sparse high-risk tail:** calibration reliability above ~0.6 predicted probability is limited by small sample size — exactly the region most clinically important to get right. A larger validation set would be needed to properly assess this.
- **Unexplained fairness gap:** the male/female AUC gap warrants further investigation, ideally with real data where the cause could be meaningfully diagnosed (e.g., different feature distributions or outcome base rates by gender) rather than left as an open question.
- **Next steps with real data:** incorporate prior admission history and days-since-last-discharge, richer comorbidity coding (e.g., Elixhauser/Charlson index) instead of a binary chronic-condition flag, and lab values at discharge.

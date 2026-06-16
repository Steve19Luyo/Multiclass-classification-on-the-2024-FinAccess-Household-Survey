# DataSprint 2026 — Predicting Financial Wellbeing in Kenya

**Strathmore Data Community (SDC) × iLab Africa**  
Multiclass classification on the 2024 FinAccess Household Survey

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Tools & Technology Stack](#tools--technology-stack)
6. [Methodology](#methodology)
7. [Exploratory Data Analysis — Key Findings](#exploratory-data-analysis--key-findings)
8. [Modeling Results](#modeling-results)
9. [Interpretation — What Drives Financial Deterioration?](#interpretation--what-drives-financial-deterioration)
10. [Policy Recommendations](#policy-recommendations)
11. [How to Reproduce](#how-to-reproduce)
12. [Deliverables](#deliverables)
13. [Limitations & Future Work](#limitations--future-work)
14. [References](#references)

---

## Executive Summary

Kenya has expanded financial inclusion dramatically over the past decade, yet **more than half of adults surveyed in 2024 reported that their financial situation had worsened** compared to the previous year. This project builds a **multiclass machine learning model** to predict whether an individual's financial status has **Improved**, **Stayed the same**, or **Worsened**, using data from **20,871 Kenyan adults** across all 47 counties.

| Item | Detail |
|---|---|
| **Task** | Multiclass classification (3 classes) |
| **Primary metric** | Weighted F1-score (handles class imbalance) |
| **Best model** | Logistic Regression (Pipeline) |
| **Test weighted F1** | **0.5298** |
| **Benchmark model** | Random Forest — weighted F1 **0.5148** |
| **Key insight** | Financial deterioration is driven less by income alone and more by **shocks, debt stress, emergency liquidity, and financial health indicators** |

---

## Problem Statement

### Background

Mobile money and digital finance have transformed Kenya, but financial wellbeing remains uneven. The 2024 FinAccess Household Survey reports that **9.9% of adults remain fully excluded** from financial services, while **52.6%** of respondents said their financial situation had **worsened** year-on-year.

### Challenge

> Using survey data from 20,871 Kenyan adults, build a model that predicts whether a person's financial situation has **Improved**, **Stayed the same**, or **Worsened**, and identify the key factors that drive financial outcomes.

### Guiding Question

> **Which factors most strongly predict financial deterioration among Kenyan adults, and what does your model suggest policymakers, banks, or NGOs should prioritise to improve financial wellbeing?**

### Target Variable — `financial_status`

| Class | Approx. share |
|---|---|
| Worsened | 52.6% |
| Stayed the same | 26.9% |
| Improved | 20.5% |

The dataset is **imbalanced** toward `Worsened`. A model that always predicts the majority class will score poorly on minority classes; **weighted F1-score** is therefore the appropriate evaluation metric.

---

## Dataset

**Source:** 2024 FinAccess Household Survey (curated for DataSprint 2026 by iLab Africa)  
**Official report:** [finaccess.knbs.or.ke](https://finaccess.knbs.or.ke/reports-and-datasets)  
**Kaggle mirror:** [Kenya FinAccess Household Survey 2024](https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024)

| Property | Value |
|---|---|
| **File** | `finaccess2024_datasprint - finaccess2024_datasprint.csv` |
| **Rows** | 20,871 |
| **Columns** | 28 (27 features + 1 target) |
| **Coverage** | All 47 Kenyan counties |

### Feature Groups

| Group | Examples |
|---|---|
| **Demographics** | `county`, `location_type`, `Sex`, `Age`, `household_size`, `education_level`, `marital_status`, `has_disability` |
| **Livelihood & income** | `monthly_income`, `experienced_shock` |
| **Mobile & digital access** | `mobile_ownership_1`, `mobile_money_access`, `barriers_mobile_money` |
| **Financial behaviour** | `Savings_formal`, `Savings_informal`, `Loan_formal`, `Loan_informal`, `defaulted`, `formal_service_use`, `barriers_bank`, `prodsum1` |
| **Financial health & literacy** | `fl_score`, `nfhi_11`, `nfhi_12`, `nfhi_13`, `accessto_13k_1month`, `not_difficult` |

### Missing Values

| Column | Missing % | Handling |
|---|---|---|
| `barriers_bank` | ~27.5% (5,734 rows) | Impute as **`"No barrier"`** — respondents without a bank account have no barrier to report |
| `monthly_income` | ~0% | Pre-imputed with median in curated dataset |

---

## Repository Structure

```
Datasprint/
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── finaccess2024_datasprint - finaccess2024_datasprint.csv   # Survey data
├── DataSprint Notebook.html                           # Full notebook export (primary)
├── Datasprint26.html                                  # Condensed notebook export
└── (deliverables to add)
    ├── finaccess_financial_status.ipynb               # Jupyter notebook
    └── DataSprint2026_Presentation.pptx               # 7-slide deck
```

---

## Tools & Technology Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **pandas** | Data loading, cleaning, inspection |
| **NumPy** | Numerical operations |
| **matplotlib** | Base plotting |
| **seaborn** | Statistical visualisations (countplot, boxplot, heatmap) |
| **scikit-learn** | Preprocessing, modelling, evaluation |
| **Jupyter Notebook** | Reproducible analysis workflow |
| **python-pptx** | Presentation generation (optional) |

### Key scikit-learn Components

| Component | Role |
|---|---|
| `train_test_split` | 80/20 stratified train/test split |
| `ColumnTransformer` | Separate numeric and categorical preprocessing |
| `StandardScaler` | Scale numeric features (zero mean, unit variance) |
| `OneHotEncoder` | Encode categorical variables |
| `Pipeline` | Prevent data leakage by chaining prep + model |
| `LogisticRegression` | Primary interpretable classifier |
| `RandomForestClassifier` | Non-linear benchmark model |
| `f1_score` | Primary evaluation metric (`average="weighted"`) |
| `classification_report` | Per-class precision, recall, F1 |
| `confusion_matrix` | Error pattern visualisation |
| `permutation_importance` | Model-agnostic feature validation |

---

## Methodology

The workflow follows the six steps required by the DataSprint brief.

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│ 1. Cleaning │ -> │  2. EDA     │ -> │ 3. Preprocessing │
└─────────────┘    └─────────────┘    └──────────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌───────▼──────────┐
│ 6. Interpret│ <- │ 5. Evaluate │ <- │  4. Modelling  │
└─────────────┘    └─────────────┘    └──────────────────┘
```

### Step 1 — Data Cleaning

1. **Load** the CSV and confirm shape `(20871, 28)`.
2. **Inspect** dtypes, missing values, and summary statistics via `df.info()`, `df.isnull().sum()`, and `df.describe()`.
3. **Impute `barriers_bank`** missing values with `"No barrier"` per competition guidance.
4. **Normalise text fields** (recommended):
   ```python
   df["education_level"] = (
       df["education_level"].astype(str)
       .str.replace('"', "", regex=False)
       .str.strip()
   )
   ```
   This removes duplicate categories caused by extra quotes and trailing spaces in the raw export.
5. **Confirm** zero missing values remain before modelling.

### Step 2 — Exploratory Data Analysis

Minimum five meaningful, labelled charts are produced:

| # | Chart | Purpose |
|---|---|---|
| 1 | Countplot of `financial_status` | Show class imbalance (majority Worsened) |
| 2 | Boxplot of `monthly_income` by `financial_status` | Compare income distributions across outcomes |
| 3 | Confusion matrix heatmap | Show where the model succeeds and fails |
| 4 | Model comparison bar chart | Compare weighted F1 across algorithms |
| 5 | Top predictors bar chart | Visualise key drivers of `Worsened` |

Each chart includes a title, axis labels, and a short written interpretation suitable for a non-technical audience.

### Step 3 — Preprocessing

**Design principle:** fit all transformations on training data only, via a single `Pipeline`, to prevent leakage.

#### Feature split

```python
num_cols = ["household_size", "monthly_income", "prodsum1"]
cat_cols = [c for c in X.columns if c not in num_cols]
```

| Type | Columns | Transformer |
|---|---|---|
| Numeric (3) | `household_size`, `monthly_income`, `prodsum1` | `StandardScaler()` |
| Categorical (24) | All remaining predictors | `OneHotEncoder(handle_unknown="ignore")` |

#### Train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

- **80% train / 20% test**
- **Stratified** to preserve class proportions
- **Fixed seed** (`random_state=42`) for reproducibility

### Step 4 — Modelling

Two classifiers are trained inside identical preprocessing pipelines:

| Model | Configuration | Rationale |
|---|---|---|
| **Logistic Regression** | `max_iter=3000`, L2 penalty | Fast, interpretable, strong baseline for survey data |
| **Random Forest** | `n_estimators=300`, `random_state=42` | Captures non-linear interactions as a benchmark |

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

lr_model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=3000, random_state=42)),
])
```

The model with the **highest weighted F1 on the test set** is selected as the final model. Logistic Regression is preferred for interpretation when scores are comparable.

### Step 5 — Evaluation

**Primary metric:**

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
```

**Supporting metrics:**

- `classification_report` — per-class precision, recall, and F1
- Confusion matrix heatmap — visual error analysis
- Side-by-side model comparison — Logistic Regression vs Random Forest

A naive baseline that always predicts `Worsened` achieves a low weighted F1 because it fails completely on `Improved` and `Stayed the same`. Our models (~0.52–0.53) demonstrate genuine predictive signal beyond the majority class.

### Step 6 — Interpretation

Interpretation uses **Logistic Regression coefficients** aligned to the `Worsened` class:

1. Extract `feature_names_out()` from the fitted preprocessor.
2. Index `clf.coef_[worsened_idx]` where `worsened_idx = list(clf.classes_).index("Worsened")`.
3. Rank features by absolute coefficient magnitude.
4. Separate **risk factors** (positive coef → higher odds of Worsened) from **protective factors** (negative coef).
5. Validate top themes with **permutation importance** on the test set.

Feature names are collapsed from one-hot encoding (e.g. `cat__experienced_shock_Yes` → `experienced_shock = Yes`) for readable reporting.

---

## Exploratory Data Analysis — Key Findings

### 1. Severe class imbalance

The majority of respondents (**~53%**) reported their financial situation **Worsened**. Only **~21%** reported improvement. Any model must perform well on all three classes, not just the majority.

### 2. Income is associated with outcome but is not deterministic

The boxplot of `monthly_income` by `financial_status` shows:

- Respondents who **Improved** tend to have a **slightly higher median income**.
- Distributions **overlap substantially** across all three classes.
- Some **high-income individuals still report Worsened** status.

**Implication:** Income alone does not explain financial trajectory. Shocks, debt, spending management, and access to emergency funds likely matter more.

### 3. Numeric feature summary

| Feature | Mean | Median | Notes |
|---|---|---|---|
| `household_size` | 4.2 | 4 | Range 1–20 |
| `monthly_income` (KES) | 9,703 | 5,000 | Highly right-skewed (max 200,000) |
| `prodsum1` | 3.9 | 3 | Count of financial products used (0–22) |

### 4. Bank barrier missingness confirms survey logic

5,734 missing `barriers_bank` values (~27.5%) correspond to respondents without bank access — correctly imputed as `"No barrier"`.

---

## Modeling Results

Evaluated on a **held-out 20% test set** (4,174 respondents) with stratified sampling.

| Model | Weighted F1 (test) | Notes |
|---|---|---|
| **Logistic Regression** | **0.5298** | Selected — best test performance + interpretable |
| Random Forest | 0.5148 | Slightly lower; less interpretable |

### Why Logistic Regression won

Although Random Forest can model complex interactions, this dataset is dominated by **categorical survey variables** after one-hot encoding. Logistic Regression captures the dominant linear patterns efficiently and supports direct coefficient-based interpretation for policymakers.

### Confusion matrix insights

The confusion matrix shows the model:

- Performs **best on `Worsened`** (largest class, most training examples).
- Struggles most with **`Improved`** and **`Stayed the same`** — the minority classes with overlapping feature profiles.
- This pattern is expected given class imbalance and the subjective nature of self-reported financial change.

---

## Interpretation — What Drives Financial Deterioration?

Based on the fitted Logistic Regression pipeline, the strongest predictors of **`Worsened`** financial status cluster into five themes:

### Theme 1 — Financial shocks and instability

| Signal | Direction |
|---|---|
| `experienced_shock = Yes` | Increases odds of Worsened |
| Income volatility (low or irregular `monthly_income`) | Associated with deterioration |

Households that experienced a financial shock in the past year are significantly more likely to report worsening finances.

### Theme 2 — Financial health indicators (NFHI)

| Signal | Direction |
|---|---|
| `nfhi_11 = No` (not food secure) | Increases odds of Worsened |
| `nfhi_12 = No` (non-food spending not managed) | Increases odds of Worsened |
| `nfhi_13 = No` (debt stress in past 3 months) | Increases odds of Worsened |

The National Financial Health Index variables are among the strongest predictors. Current financial stress is a direct indicator of perceived year-on-year deterioration.

### Theme 3 — Emergency liquidity gap

| Signal | Direction |
|---|---|
| `accessto_13k_1month = No` | Increases odds of Worsened |
| `not_difficult = No` (emergency funds hard to access) | Increases odds of Worsened |

Inability to access KES 13,000 within 30 days — the survey's resilience benchmark — strongly predicts financial deterioration.

### Theme 4 — Debt and loan distress

| Signal | Direction |
|---|---|
| `defaulted = Yes` | Increases odds of Worsened |
| Informal loan usage without formal savings | Associated with higher risk |

Defaulting on a loan is a clear marker of financial distress and strongly linked to worsening status.

### Theme 5 — Financial access barriers and low product usage

| Signal | Direction |
|---|---|
| Low `prodsum1` (few financial products used) | Associated with Worsened |
| `barriers_bank = Affordability` | Associated with Worsened |
| Mobile money / phone access barriers | Associated with Worsened |
| `formal_service_use = Non-usage` | Associated with Worsened |

Households excluded from or unable to afford formal financial services face higher deterioration risk.

### Protective factors (associated with Improved / not Worsened)

| Signal | Direction |
|---|---|
| `nfhi_11/12/13 = Yes` (food secure, managed spending, no debt stress) | Protective |
| `accessto_13k_1month = Yes` | Protective |
| Higher `fl_score` (financial literacy) | Protective |
| Active formal/informal savings usage | Protective |
| Higher `prodsum1` (diverse product usage) | Protective |

---

## Policy Recommendations

The following recommendations are derived directly from model findings and EDA. Each maps to a measurable feature in the survey.

### 1. Build shock-responsive safety nets

**Evidence:** `experienced_shock = Yes` is a top predictor of deterioration.

**Action:** Design **rapid-response cash or voucher programmes** triggered by drought, medical emergencies, or job loss — especially in rural counties with high shock exposure. Partner with mobile money providers for last-mile delivery within 48 hours of a verified shock event.

**Stakeholders:** National government (social protection), NGOs (humanitarian cash transfers), M-Pesa / mobile operators.

---

### 2. Expand emergency liquidity products

**Evidence:** `accessto_13k_1month = No` and `not_difficult = No` strongly predict Worsened status.

**Action:** Develop **affordable emergency savings and micro-insurance products** targeting the KES 13,000 resilience threshold. Simplify onboarding (USSD-based enrolment, no minimum balance requirements) and offer matched savings incentives for low-income households.

**Stakeholders:** Banks, SACCOs, microfinance institutions, CBK (regulatory sandbox).

---

### 3. Address affordability barriers to banking

**Evidence:** `barriers_bank = Affordability` is the most common reported barrier; it correlates with deterioration.

**Action:** Introduce **zero-fee basic bank accounts** and **tiered pricing** for digital transactions in underserved counties. Subsidise account maintenance for households below the median income (KES 5,000/month).

**Stakeholders:** Commercial banks, CBK, FSD Kenya.

---

### 4. Scale debt counselling and restructuring programmes

**Evidence:** `defaulted = Yes` and `nfhi_13 = No` (debt stress) are strong deterioration signals.

**Action:** Fund **community-based debt advisory services** linked to informal lending groups (chamas) and formal lenders. Offer structured repayment plans before defaults occur, with referral pathways from NFHI screening questions.

**Stakeholders:** Banks, MFIs, county governments, financial literacy NGOs.

---

### 5. Close the digital and eligibility gap

**Evidence:** Barriers related to phone ownership, eligibility, and awareness (`barriers_mobile_money`, `barriers_bank = Eligibility/Awareness`) predict exclusion and deterioration.

**Action:** Run **national ID registration drives** in counties with high eligibility barriers. Deploy **mobile money agent networks** in rural areas and provide subsidised smartphones tied to financial literacy training (`fl_score` improvement programmes).

**Stakeholders:** NGOs, mobile network operators, county administrations, KNBS / registration bodies.

---

### 6. Invest in financial literacy tied to product usage

**Evidence:** Higher `fl_score` and `prodsum1` are protective; low literacy and zero product usage predict Worsened.

**Action:** Integrate **contextual financial education** (budgeting, savings, debt management) into existing group lending and mobile money onboarding flows. Track `prodsum1` as a programme outcome metric.

**Stakeholders:** FSD Kenya, SACCOs, schools, community organisations.

---

## How to Reproduce

### Prerequisites

- Python 3.10 or later
- pip

### Setup

```bash
cd Datasprint
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook
```

Open `finaccess_financial_status.ipynb` (or the HTML exports `DataSprint Notebook.html` for reference) and run all cells sequentially.

### Expected outputs

| Output | Expected value |
|---|---|
| Dataset shape | `(20871, 28)` |
| Missing values after cleaning | 0 |
| Test weighted F1 (Logistic Regression) | ~0.53 |
| Test weighted F1 (Random Forest) | ~0.51 |
| Charts generated | ≥ 5 |

---

## Deliverables

| Deliverable | Status | Description |
|---|---|---|
| Jupyter Notebook (`.ipynb`) | To export | Full pipeline: cleaning → EDA → prep → model → evaluation → interpretation |
| HTML notebook export | Done | `DataSprint Notebook.html` |
| Data visualisations (≥ 5) | Done | Included in notebook |
| 7-slide PowerPoint (`.pptx`) | To create | Problem, Dataset & EDA, Key finding, Modelling, Performance, Key drivers, Recommendations |
| README | Done | This file |

### Suggested slide mapping

| Slide | Content source in this README |
|---|---|
| 1 — Problem | [Problem Statement](#problem-statement) |
| 2 — Dataset & EDA | [Dataset](#dataset) + [EDA Findings](#exploratory-data-analysis--key-findings) |
| 3 — Key EDA finding | Income boxplot insight (Section EDA #2) |
| 4 — Modelling approach | [Methodology Steps 3–4](#step-3--preprocessing) |
| 5 — Model performance | [Modeling Results](#modeling-results) |
| 6 — Key drivers | [Interpretation](#interpretation--what-drives-financial-deterioration) |
| 7 — Recommendations | [Policy Recommendations](#policy-recommendations) |

---

## Limitations & Future Work

| Limitation | Impact | Mitigation |
|---|---|---|
| **Self-reported target** | `financial_status` is subjective; no ground-truth validation | Complement with objective NFHI indicators (already used as features) |
| **Class imbalance** | Model biased toward `Worsened`; minority classes harder to predict | Consider `class_weight="balanced"`, SMOTE, or cost-sensitive learning |
| **Cross-sectional data** | Cannot establish causality — only associations | Frame findings as predictive, not causal |
| **Contemporaneous features** | NFHI variables measured in the same period as the target | Acknowledge potential overlap; consider a features-only model excluding NFHI for robustness |
| **No hyperparameter tuning** | Default model settings may underperform | Grid search on `C` (LR) or `max_depth` / `n_estimators` (RF) with cross-validation |
| **Income skewness** | `monthly_income` is right-skewed | Apply `log1p` transform before scaling |
| **Education label noise** | Duplicate categories from CSV quoting | Strip quotes and whitespace before encoding |

### Future improvements

- **Hyperparameter optimisation** via `GridSearchCV` with stratified k-fold
- **Alternative models**: Gradient Boosting (XGBoost/LightGBM) for higher accuracy
- **SHAP values** for individual-level explanations
- **County-level aggregation** for geospatial policy targeting
- **Temporal modelling** if longitudinal FinAccess waves become available

---

## References

1. Central Bank of Kenya, KNBS, and FSD Kenya. *FinAccess Household Survey 2024.* [finaccess.knbs.or.ke](https://finaccess.knbs.or.ke/reports-and-datasets)
2. Strathmore Data Community. *DataSprint 2026 Problem Statement.* SDC × iLab Africa.
3. scikit-learn documentation: [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
4. Kaggle dataset: [Kenya FinAccess Household Survey 2024](https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024)

---

## Contact

**Strathmore Data Community**  
datacommunity@strathmore.edu | [linktr.ee/strathmoredatacommunity](https://linktr.ee/strathmoredatacommunity)

---

*Built for DataSprint 2026 — Learn. Build. Launch.*

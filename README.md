# Insurance Risk Analytics

A data-driven analytics project analyzing car insurance policy, client, vehicle,
and claim data for **ACIS (African Car Insurance Solutions)** covering
February 2014 to August 2015.

---

## Business Objective

ACIS is reviewing its historical data to optimize its marketing strategy and
identify low-risk customer segments for premium reduction opportunities.
This project delivers insights across four analytical stages:

- **Descriptive** What happened? (EDA, loss ratios, claim distributions)
- **Diagnostic** Why did it happen? (risk drivers, geographic trends)
- **Predictive** What will happen? (claim probability and severity modeling)
- **Prescriptive** What should we do? (pricing and marketing recommendations)

---

## Project Structure

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py
├── .dvc/
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains car insurance policy, client, vehicle, and claim
information with the following key groups:

| Group | Key Fields |
|---|---|
| Policy | UnderwrittenCoverID, PolicyID |
| Client | Gender, MaritalStatus, Language, Bank |
| Location | Province, PostalCode, MainCrestaZone |
| Vehicle | VehicleType, make, RegistrationYear, CustomValueEstimate |
| Plan | SumInsured, TotalPremium, CoverType |
| Claims | TotalClaims |

### Key Derived Metrics

- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims
- **Claim Rate** = Percentage of policies with at least one claim

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Maki4444444/Insurance_Risk_Analytics_.git
cd Insurance_Risk_Analytics_
```

### 2. Create and Activate a Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the raw dataset inside:

```text
data/raw/MachineLearningRating_v3.txt
```

### 5. Run the Notebooks

#### Exploratory Data Analysis

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

#### Hypothesis Testing

```powershell
jupyter notebook notebooks/02_hypothesis_testing.ipynb
```

#### Predictive Modeling

```powershell
jupyter notebook notebooks/03_modeling.ipynb
```

---

## CI Pipeline

This project uses GitHub Actions for continuous integration.

On every push and pull request, the pipeline automatically:

- Lints code using `flake8`
- Runs tests using `pytest`

Pipeline status is available in the GitHub **Actions** tab.

---

## Data Version Control (DVC)

This project uses DVC to version datasets and ensure full reproducibility.

### Pull the Dataset

```bash
dvc pull
```

### Data Versions Tracked

| Version | File | Description |
|---|---|---|
| v1 | `data/raw/MachineLearningRating_v3.txt` | Original raw dataset |
| v2 | `data/processed/insurance_data_cleaned.csv` | Cleaned dataset after preprocessing |

### Switch Between Dataset Versions

```bash
git checkout <commit-hash>
dvc checkout
```

---

## Error Handling

The project implements defensive programming practices including:

- File existence validation
- CSV parsing exception handling
- Missing column checks
- Visualization error handling
- Safe execution wrappers for analysis scripts

---

# Task 1: Exploratory Data Analysis (EDA)

## Objective

Perform exploratory analysis to understand portfolio profitability,
claim behavior, customer segmentation, and geographic risk trends.

---

## Key Findings (Task 1)

| Question | Finding |
|---|---|
| Overall Loss Ratio | 1.0477 — portfolio is currently unprofitable |
| Highest risk provinces | Gauteng (1.22), KwaZulu-Natal (1.08), Western Cape (1.06) |
| Lowest risk provinces | Northern Cape (0.28), Eastern Cape (0.63), Limpopo (0.66) |
| Highest risk vehicle type | Heavy Commercial (loss ratio 1.63) |
| Most profitable vehicle type | Bus (0.14), Light Commercial (0.23) |
| Highest avg claim make | Suzuki (~420), JMC (~190), Hyundai (~165) |
| Gender with lowest loss ratio | Female (0.82) most profitable segment |
| Temporal trend | Claims exceeded premiums consistently from Oct 2014 onwards |

---

## Strategic Recommendations (EDA)

- Reprice high-risk provinces immediately
- Review underwriting for Heavy Commercial vehicles
- Target profitable customer segments for acquisition
- Expand retention strategies for low-risk policyholders

---

# Task 2: Data Version Control (DVC)

## Objective

Implement reproducible data pipelines using DVC for:

- Dataset versioning
- Pipeline reproducibility
- Auditability
- Collaboration

---

## DVC Workflow

### Track Dataset

```bash
dvc add data/raw/MachineLearningRating_v3.txt
```

### Push Dataset to Remote Storage

```bash
dvc push
```

### Reproduce Pipeline

```bash
dvc repro
```

---

# Task 3: A/B Hypothesis Testing

## Objective

Evaluate whether statistically significant differences exist between
customer groups, vehicle categories, and geographic regions.

The analysis supports pricing optimization and underwriting decisions.

---

## Statistical Tests Performed

| Test | Purpose |
|---|---|
| Province vs Claim Frequency | Compare claim occurrence by region |
| Gender vs Claim Frequency | Evaluate gender-based differences |
| Vehicle Type vs Loss Ratio | Compare vehicle risk categories |
| Postal Code vs Severity | Evaluate geographic severity patterns |
| Margin Analysis | Compare profitability between segments |

---

## Methodology

The hypothesis testing workflow includes:

- Null and alternative hypothesis formulation
- Statistical significance testing
- P-value interpretation
- Confidence interval analysis
- Group mean comparison

### Techniques Used

- Independent t-tests
- Chi-square tests
- Loss ratio analysis
- Group comparison statistics

---

## Key Findings (Task 3)

| Hypothesis | Result |
|---|---|
| Claim frequency differs by province | Confirmed |
| Heavy Commercial vehicles are riskier | Confirmed |
| Geographic regions show distinct risk profiles | Confirmed |
| Gender impact exists but is relatively weak | Observed |
| Some low-risk segments appear overpriced | Likely |

---

## Business Impact (Task 3)

The hypothesis testing phase provides statistical evidence for:

- Risk-based pricing adjustments
- Province-specific premium strategies
- Vehicle-category underwriting improvements
- Better actuarial segmentation
- More targeted marketing campaigns

---

# Task 4: Predictive Modeling

## Objective

Develop predictive machine learning models for:

1. Claim Probability Prediction (Classification)
2. Claim Severity Prediction (Regression)

These models support expected-loss pricing and portfolio risk management.

---

## Feature Engineering

The modeling pipeline introduced several engineered features:

| Feature | Description |
|---|---|
| VehicleAge | Derived from registration year |
| Margin | Premium minus claims |
| HasClaim | Binary claim indicator |
| PremiumToSumInsuredRatio | Relative premium intensity |

---

## Data Preparation

### Cleaning & Preprocessing

- Missing value imputation
- Categorical encoding
- Numeric feature selection
- Outlier handling
- Log transformation for severity modeling

### Final Dataset

| Metric | Value |
|---|---|
| Total rows | 1,000,098 |
| Final features | 36 |
| Remaining null values | 0 |

---

## Claim Severity Analysis

| Metric | Value |
|---|---|
| Policies with claims | 2,788 |
| Mean claim | R23,273 |
| Median claim | R6,140 |
| Maximum claim | R393,092 |
| Skewness | 3.85 |

The claim severity distribution is heavily right-skewed,
justifying the use of log transformation.

---

## Class Imbalance

| Metric | Value |
|---|---|
| Claim rate | 0.2788% |
| Imbalance ratio | 358:1 |

The dataset is extremely imbalanced, making accuracy an unreliable metric.
Special imbalance handling techniques were required during classification.

---

## Classification Modeling

### Models Evaluated

| Model |
|---|
| Logistic Regression |
| Random Forest |
| XGBoost |

### Classification Results

| Model | Accuracy | Recall | AUC |
|---|---|---|---|
| XGBoost | 0.8455 | 0.9068 | 0.9249 |
| Logistic Regression | 0.7894 | 0.7724 | 0.8554 |
| Random Forest | 0.9706 | 0.1935 | 0.6226 |

### Classification Insight

XGBoost achieved the strongest overall performance with excellent claim
detection capability and strong ranking power across high-risk policies.

---

## Regression Modeling

### Models Evaluated

| Model |
|---|
| Linear Regression |
| Decision Tree |
| Random Forest |
| XGBoost |

### Regression Results

| Model | RMSE | R² |
|---|---|---|
| Tuned Random Forest | R36,318.57 | 0.6510 |
| XGBoost | R36,521.75 | 0.6267 |
| Linear Regression | R36,548.12 | 0.6259 |
| Decision Tree | R44,451.72 | 0.4938 |

### Regression Insight

The tuned Random Forest model delivered the strongest overall regression
performance while maintaining good generalization stability.

---

## Cross-Validation Results

| Model | CV R² |
|---|---|
| XGBoost | 0.6528 ± 0.0156 |
| Random Forest | 0.6317 ± 0.0262 |
| Decision Tree | 0.5696 ± 0.0303 |

XGBoost demonstrated the most stable generalization performance across folds.

---

## Hyperparameter Tuning

### Best Random Forest Parameters

```python
{
    'bootstrap': True,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 200
}
```

### Tuning Impact

| Metric | Untuned RF | Tuned RF |
|---|---|---|
| RMSE | R36,243.45 | R36,318.57 |
| R² | 0.6166 | 0.6510 |

The tuned model improved generalization while reducing overfitting.

---

## Final Business Recommendations

### Pricing Strategy

- Reprice high-risk provinces
- Adjust Heavy Commercial vehicle premiums
- Incorporate predicted claim probability into pricing

### Marketing Strategy

- Target profitable low-risk customer segments
- Expand acquisition in low-loss regions
- Improve retention for profitable customers

### Operational Strategy

- Monitor extreme claims separately
- Retrain models periodically
- Continue collecting richer behavioral data

---

## Future Improvements

Potential next steps include:

- SMOTE imbalance handling
- SHAP explainability analysis
- Probability calibration
- Time-series trend modeling
- API deployment for real-time scoring

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable production-ready code |
| `task-1` | EDA and project setup |
| `task-2` | DVC implementation |
| `task-3` | Hypothesis testing |
| `task-4` | Predictive modeling |


## Insurance Risk Analytics

A data-driven analytics project analyzing car insurance policy, client, vehicle,
and claim data for **ACIS (African Car Insurance Solutions)** covering
February 2014 to August 2015.

---

### Business Objective

ACIS is reviewing its historical data to optimize its marketing strategy and
identify "low-risk" targets for premium reduction opportunities. This project
delivers insights across four analytical stages:

- **Descriptive** What happened? (EDA, loss ratios, claim distributions)
- **Diagnostic** Why did it happen? (risk drivers, geographic trends)
- **Predictive** What will happen? (claim probability and severity modeling)
- **Prescriptive** What should we do? (pricing and marketing recommendations)

---

### Project Structure
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── data/
│   └── raw/                    # Raw dataset (tracked by DVC, not Git)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── init.py
│   ├── data_loader.py          # Data loading utilities
│   ├── eda_utils.py            # EDA helper functions and plotting
│   ├── hypothesis_tests.py     # Statistical testing functions
│   └── modeling.py             # ML modeling functions
├── reports/
│   └── final_report.md         # Final business report
├── tests/
│   ├── init.py
│   └── test_placeholder.py
├── .dvc/                       # DVC configuration
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # Python dependencies
└── README.md
---

### Dataset

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

**Key derived metrics:**
- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims
- **Claim Rate** = % of policies with at least one claim

---

### Setup & Installation

#### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/insurance_risk_analytics_.git
cd insurance_risk_analytics_
```

#### 2. Create and activate a virtual environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Add the dataset
Place the raw dataset in `data/raw/`:
data/raw/MachineLearningRating_v3.txt

#### 5. Run the EDA notebook
```powershell
jupyter notebook notebooks/01_eda.ipynb
```

---

### CI Pipeline

This project uses **GitHub Actions** for continuous integration. On every
push and pull request, the pipeline automatically:

- Lints the code with `flake8`
- Runs tests with `pytest`

Pipeline status is visible under the **Actions** tab on GitHub.

---
### Data Version Control (DVC)

This project uses DVC to version and reproduce the data pipeline,
ensuring every analysis result is reproducible and auditable.

#### Reproduce the pipeline

**1. Pull the data from remote storage:**
```bash
dvc pull
```

**2. Data versions tracked:**

| Version | File | Description |
|---|---|---|
| v1 | `data/raw/MachineLearningRating_v3.txt` | Original raw dataset (503MB) |
| v2 | `data/processed/insurance_data_cleaned.csv` | Cleaned dataset after missing value handling |

**3. Switch between versions:**
```bash
git checkout <commit-hash>
dvc checkout
```
### Error Handling

The project implements defensive programming practices including:

- File existence validation
- CSV parsing exception handling
- Missing column checks
- Visualization error handling
- Safe execution wrappers for analysis scripts

### Key Findings (Task 1: EDA)

| Question | Finding |
|---|---|
| **Overall Loss Ratio** | 1.0477 — portfolio is currently unprofitable |
| **Highest risk provinces** | Gauteng (1.22), KwaZulu-Natal (1.08), Western Cape (1.06) |
| **Lowest risk provinces** | Northern Cape (0.28), Eastern Cape (0.63), Limpopo (0.66) |
| **Highest risk vehicle type** | Heavy Commercial (loss ratio 1.63) |
| **Most profitable vehicle type** | Bus (0.14), Light Commercial (0.23) |
| **Highest avg claim make** | Suzuki (~420), JMC (~190), Hyundai (~165) |
| **Gender with lowest loss ratio** | Female (0.82) most profitable segment |
| **Temporal trend** | Claims exceeded premiums consistently from Oct 2014 onwards |

---

### Strategic Recommendations

- **Gauteng, KwaZulu-Natal, Western Cape** require immediate premium repricing
- **Heavy Commercial vehicles** premiums should be reviewed and risk controls strengthened
- **Suzuki and JMC** underwriting guidelines should be reviewed
- **Northern Cape and Light Commercial** are ideal targets for customer acquisition
- **Female policyholders** are a strong segment for loyalty and discount programs

---

### Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code |
| `task-1` | EDA and project setup |
| `task-2` | Data Version Control (DVC) |
| `task-3` | A/B Hypothesis Testing |
| `task-4` | Statistical Modeling |



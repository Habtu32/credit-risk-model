# Credit Risk Scoring – Interim Report (Task 1 & Task 2)

## Project Overview
This project focuses on building a credit risk scoring framework using transaction-level financial data. The goal is to understand customer behavior, explore data characteristics, and prepare the foundation for interpretable credit risk models that align with regulatory standards such as the Basel II Accord.

The interim phase of the project covers:
- Business understanding of credit risk modeling
- Exploratory Data Analysis (EDA)

---

## Credit Scoring Business Understanding

### Regulatory Context: Basel II Accord
The Basel II Accord emphasizes accurate measurement and management of credit risk to ensure financial stability. Financial institutions are required to justify their credit decisions using transparent, auditable, and well-documented models. As a result, model interpretability is a key requirement, especially when models influence lending decisions, capital allocation, and regulatory reporting.

### Need for a Proxy Default Variable
The dataset does not contain a direct indicator of customer default. To address this limitation, a **proxy variable** is required to approximate risky or fraudulent behavior using observable transaction patterns.

However, using proxy variables introduces business risks:
- The proxy may not perfectly represent true default behavior.
- Customers may manipulate transaction behavior if proxy logic becomes known.
- Incorrect proxy definitions may lead to biased or unfair credit decisions.

Therefore, proxy variables must be carefully designed, validated, and documented.

### Model Interpretability vs Predictive Power
In regulated financial environments, there is a trade-off between:
- **Simple, interpretable models** (e.g., Logistic Regression with Weight of Evidence)
- **Complex, high-performance models** (e.g., Gradient Boosting)

While complex models may achieve higher predictive accuracy, they are harder to explain and audit. Simpler models are easier to interpret, debug, and justify to regulators. Given these constraints, this project prioritizes **model explainability**, even if it slightly reduces accuracy.

---

## Exploratory Data Analysis (EDA)

### Dataset Overview
- Total records: 95,662
- Total features: 16
- Data type: Transaction-level financial data

### Numerical Feature Analysis
- Transaction `Amount` and `Value` are highly right-skewed.
- Extreme outliers exist, with very large transaction values.
- Strong correlation is observed between `Amount` and `Value`, indicating redundancy.

### Categorical Feature Analysis
- Features such as `ProductCategory` and `ChannelId` are highly imbalanced.
- A small number of categories dominate transaction volume.
- Rare categories may require grouping or special handling during modeling.

### Correlation Analysis
- A strong positive correlation exists between `Amount` and `Value`.
- Both features show moderate correlation with the fraud-related outcome variable.

### Missing Values
- No missing values were detected in the dataset.
- This simplifies preprocessing and removes the need for imputation.

### Outlier Detection
- Significant outliers were identified in numerical features.
- These outliers may influence model performance and require robust handling techniques.

### Key EDA Insights
1. The dataset is transaction-level and requires aggregation to customer-level features.
2. Numerical features are skewed with extreme outliers.
3. Certain numerical variables are highly correlated.
4. Categorical variables are imbalanced.
5. No missing values were found.

---

## Repository Structure
├── notebooks/
│ └── eda.ipynb
├── src/
├── tests/
├── data/ (ignored via .gitignore)
├── requirements.txt
├── .gitignore
└── README.md

## Setup Instructions
## Setup Instructions
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
```bash
pip install -r requirements.txt
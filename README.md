# Credit Scoring Business Understanding.

Basel II places strong emphasis on accurate risk measurement and regulatory compliance, which requires banks to use models that are interpretable and well documented. An interpretable model allows the bank to explain credit decisions to regulators, auditors, and customers, providing clear evidence of how risk is assessed. Without transparency and documentation, even a highly accurate model may not be acceptable in a regulated financial environment.

## why a proxy is needed
Since the dataset does not contain a direct default label, creating a proxy variable is necessary to estimate credit risk. This proxy is derived from customer behavioral engagement patterns, such as transaction frequency and monetary value, which serve as indirect indicators of repayment likelihood. However, using a proxy introduces business risks because it is based on assumptions rather than actual default outcomes. Customers may behave differently than expected, and in some cases, users could manipulate their activity to appear low-risk, leading to incorrect credit decisions and potential financial loss.

In a regulated financial environment, there is a trade-off between model interpretability and predictive performance. Simple models such as Logistic Regression with Weight of Evidence are easier to interpret, validate, and explain to regulators, auditors, and business stakeholders. This transparency makes it easier to identify errors, biases, and model weaknesses. In contrast, complex models like Gradient Boosting often achieve higher accuracy but act as black boxes, making them harder to explain and riskier to deploy in compliance-driven banking environments.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run API: `uvicorn src.api.main:app --reload`

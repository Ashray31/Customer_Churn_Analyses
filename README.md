# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their service, using the IBM Telco Customer Churn dataset.

## Business Problem

Acquiring a new customer costs far more than keeping an existing one. If a company can flag customers who are likely to leave *before* they leave, it can step in with retention offers, support calls, or better pricing — and protect revenue. This project builds a simple, interpretable model that scores each customer's risk of churning.

## Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` (Yes/No) — about 27% of customers in the dataset churned

Features include customer demographics (gender, senior citizen status, partner/dependents), account details (tenure, contract type, payment method), and services signed up for (internet, phone, streaming, tech support, etc.).

## Approach

1. **Data cleaning** — checked for missing values, converted `TotalCharges` to a numeric column and filled a small number of missing entries with the median.
2. **Exploratory analysis** — looked at the churn distribution to understand class balance (roughly 3-to-1, No vs Yes).
3. **Encoding** — converted categorical columns (contract type, payment method, services, etc.) into numeric form.
4. **Train/test split** — held out 20% of customers to test on data the model hadn't seen.
5. **Feature scaling** — standardized numeric features.
6. **Modeling** — trained a Random Forest classifier to predict churn.
7. **Evaluation** — measured accuracy and reviewed a confusion matrix to see what kinds of mistakes the model makes.

## Results

- **Accuracy:** 78% on the test set
- The confusion matrix shows the model is noticeably better at catching customers who *stay* than customers who *churn* — expected, given churners are the minority class (~27% of the data). This is a natural next area to improve (see below).

## What I'd Improve Next

- Address class imbalance (e.g. class weighting or resampling) so the model catches more actual churners, not just overall accuracy
- Look at precision/recall/F1 per class, not just accuracy, since accuracy can be misleading on imbalanced data
- Try other models (Logistic Regression, XGBoost) and compare
- Add feature importance / explainability so the "why" behind each prediction is clear

## Project Structure

```
├── customer_churn.ipynb   # Full analysis: EDA, preprocessing, modeling, evaluation
├── requirements.txt       # Python dependencies
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook customer_churn.ipynb
```

You'll need the [Telco Customer Churn CSV](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) placed in the same folder (or update the file path in the first cell).

## Tools Used

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib

# Telco Customer Churn Prediction

**Why do telecom companies lose customers? Can we predict it before it happens?**  
This project answers that question by building a full **machine learning pipeline** to forecast **customer churn** using demographic, service, and billing data.

It’s not just about accuracy — it’s about showing **how a raw business problem becomes a deployable ML solution.**

---

## What This Project Shows

- How to take **raw messy data** and turn it into a clean, engineered dataset.
- How to handle **class imbalance** in real-world churn problems.
- How to compare **two advanced models** — XGBoost and CatBoost — and decide which works best.
- How to **explain results** to non-technical stakeholders (with feature importance plots, churn distributions, etc.).
- How to make the project **reusable & demo-ready** with a single command (`run_all.py`).

---

## Workflow in 5 Steps

1. **Data Preprocessing** → Clean missing values, encode categories, create new features like `AvgMonthlySpend` & `TenureGroup`.

   - Saved as `processed_churn.csv` (34 engineered features).
   - All `.py` scripts use this file directly for efficiency.
   - You can also re-generate it interactively in the Notebook.

2. **Exploratory Data Analysis (EDA)** → Visualize churn patterns (tenure, charges, contract types).

3. **Model Training**

   - **XGBoost** (fast, strong baseline).
   - **CatBoost** (handles categorical features natively).

4. **Evaluation** → Compare Accuracy, Precision, Recall, F1, ROC AUC, PR curves.

5. **Live Demo** → Predict churn for a single new customer with one line of code.

---

## Results That Matter

- **CatBoost slightly outperformed XGBoost**:

  - Accuracy: **76.1%** vs 76.9%
  - Recall (churn detection): **0.71** vs 0.71
  - ROC AUC: **0.83** vs 0.83

- **Key Insights**:

  - Customers with **short tenure** and **high monthly charges** are far more likely to churn.
  - Long-tenure customers with high total spend tend to stay loyal.

- **Visuals Produced**: churn distribution, feature importance, ROC/PR curves, and churn vs charges.

---

## What’s Inside

```
Telco-Customer-Churn-Prediction/
│── Data/              # raw + processed dataset
│── src/               # modular pipeline scripts
│── models/            # saved CatBoost & XGBoost models
│── Results/           # evaluation plots
│── run_all.py         # runs the entire pipeline
│── Notebooks/         # interactive analysis & training
│── requirements.txt   # dependencies
│── README.md
```

---

## Project Configuration Guide

Here are the main ways you can experiment with the project.  
All modifications can be made in the **`.py` scripts under `src/`** or directly in the **Notebook (`.ipynb`)** for a more interactive experience.

- **Demo toggle (`run_all.py`):**

  - `include_demo = True` → run the demo (`src/10_demo.py`).
  - `include_demo = False` → skip the demo.

- **Model training flags (`src/05_xgboost.py` & `src/06_catboost.py`):**

  - `use_pretrained_xgb = True/False` → load or train XGBoost model.
  - `use_pretrained_cat = True/False` → load or train CatBoost model.
  - These flags can also be adjusted inside the Notebook to control training interactively.

- **Live demo customer index (`src/10_demo.py`):**
  - Change `sample = X_test.iloc[0]` to another index.
  - Example: `sample = X_test.iloc[42]` → predicts for the 43rd customer.
  - ⚠️ Valid indices: 0 to 1408 (1,409 test customers).
  - In the Notebook, you can do the same with one line of code and instantly see results.

Reminder: Save the file (or Notebook cell) after making changes before re-running.

---

## How to Run It

```bash
git clone https://github.com/RohitSaiKiran/Predicting-Customer-Churn-XGBoost-CatBoost
cd Predicting-Customer-Churn-XGBoost-CatBoost
pip install -r requirements.txt
python run_all.py
```

- Run predictions interactively → `src/10_demo.py`.
- Or experiment in the Notebook for step-by-step exploration.

---

## Project Impact

- Streamlined churn prediction workflow into a single reusable pipeline.
- Highlighted factors driving churn to support retention strategies.
- Evaluated and benchmarked two industry-grade ML models.
- Designed the project for reproducibility, scalability, and clarity.

---

## Author

**Rohit Sai Kiran Ravula**  
rohitsaikiran.r@gmail.com  
[GitHub](https://github.com/RohitSaiKiran)

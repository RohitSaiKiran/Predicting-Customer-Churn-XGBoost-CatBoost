"""
03_eda.py
---------
Exploratory Data Analysis (EDA) for churn dataset.
- Distribution of churned vs non-churned customers
- Distribution of numerical features
- Monthly charges compared by churn status
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
df = pd.read_csv("Data/processed_churn.csv")

# Churn distribution
sns.countplot(x="Churn", data=df, hue="Churn", palette="Set2", legend=False)
plt.title("Churn Distribution", fontsize=14, weight="bold")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

# Numerical features distribution
df[["tenure", "MonthlyCharges", "TotalCharges"]].hist(
    bins=30, figsize=(12,6), grid=False, color="#4C72B0", edgecolor="black"
)
plt.suptitle("Numerical Features Distribution", fontsize=14, weight="bold")
plt.show()

# Churn vs Monthly Charges
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, hue="Churn", palette="Set2", legend=False)
plt.title("Monthly Charges vs Churn", fontsize=14, weight="bold")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges ($)")
plt.show()
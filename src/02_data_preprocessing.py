"""
02_data_preprocessing.py
------------------------
Data loading, cleaning, encoding, and feature engineering for churn dataset.
"""

import pandas as pd

# Load dataset
df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature Engineering (same as notebook)
df["AvgMonthlySpend"] = df.apply(lambda x: x["TotalCharges"]/x["tenure"] if x["tenure"] > 0 else 0, axis=1)

def tenure_group(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 36:
        return "13-36"
    else:
        return "37-72"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Save processed dataset
df.to_csv("Data/processed_churn.csv", index=False)

print(df.head())
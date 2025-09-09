"""
10_demo.py
----------
Single-customer churn prediction demo.
Loads trained XGBoost and CatBoost models and predicts churn for one test row.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load preprocessed dataset
df = pd.read_csv("Data/processed_churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load models
xgb_model = XGBClassifier()
xgb_model.load_model("models/xgb_model.json")

cat_model = CatBoostClassifier()
cat_model.load_model("models/cat_model.cbm")

# Single-customer prediction function
def predict_customer(model, row, model_name):
    row = row.values.reshape(1, -1)
    pred = model.predict(row)[0]
    result = "Churn" if pred == 1 else "No Churn"
    print(f"{model_name} Prediction:", result)

# Example: predict for the first test row
sample = X_test.iloc[0]   # Try changing the index (0 - len(X_test)) to test different customers
predict_customer(xgb_model, sample, "XGBoost")
predict_customer(cat_model, sample, "CatBoost")
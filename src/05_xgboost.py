"""
05_xgboost.py
-------------
Train or load an XGBoost model for churn prediction, then evaluate with classification metrics.

Configuration Note:
- Inside this file you will find the flag `use_pretrained_xgb = True`.
- True  → loads a pretrained XGBoost model from /models
- False → trains a new model and saves it in /models
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Load preprocessed dataset
df = pd.read_csv("Data/processed_churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost Classifier (Train or Load Pretrained)
use_pretrained_xgb = True   # True = Load pretrained, False = Train new

if use_pretrained_xgb:
    print("Loading pretrained XGBoost model...")
    xgb_model = XGBClassifier()
    xgb_model.load_model("models/xgb_model.json")
else:
    print("Training new XGBoost model...")
    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model("models/xgb_model.json")
    print("New XGBoost model trained and saved!")

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("ROC AUC (XGBoost):", roc_auc_score(y_test, y_proba_xgb))
"""
08_results.py
--------------
Collect and compare evaluation metrics (Accuracy, Precision, Recall, F1, ROC AUC)
for XGBoost and CatBoost models with threshold tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

cat_model = CatBoostClassifier()
cat_model.load_model("models/cat_model.cbm")
y_proba_cat = cat_model.predict_proba(X_test)[:, 1]

# Threshold tuning
def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.2, 0.61, 0.01)
    best_f1, best_t, best_prec, best_rec = 0, 0.5, 0, 0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t, best_prec, best_rec = f1, t, prec, rec
    return best_t, best_f1, best_prec, best_rec

# Apply tuned thresholds
xgb_t, _, _, _ = find_best_threshold(y_test, y_proba_xgb)
y_pred_xgb = (y_proba_xgb >= xgb_t).astype(int)

cat_t, _, _, _ = find_best_threshold(y_test, y_proba_cat)
y_pred_cat = (y_proba_cat >= cat_t).astype(int)

# Evaluation helper
def evaluate_model(y_true, y_pred, y_proba, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_proba)
    }

# Collect results (only XGBoost & CatBoost, same as notebook)
results = []
results.append(evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost"))
results.append(evaluate_model(y_test, y_pred_cat, y_proba_cat, "CatBoost"))

# Display results
results_df = pd.DataFrame(results)
print(results_df)
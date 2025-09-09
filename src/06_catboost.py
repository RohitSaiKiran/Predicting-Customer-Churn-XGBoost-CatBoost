"""
06_catboost.py
--------------
Train or load a CatBoost model for churn prediction, then evaluate performance.

Configuration Note:
- Inside this file you will find the flag `use_pretrained_cat = True`.
- True  → loads a pretrained CatBoost model from /models
- False → trains a new model and saves it in /models
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
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

# CatBoost Classifier (Train or Load Pretrained)
use_pretrained_cat = True   # True = Load pretrained, False = Train new

if use_pretrained_cat:
    print("Loading pretrained CatBoost model...")
    cat_model = CatBoostClassifier()
    cat_model.load_model("models/cat_model.cbm")
else:
    print("Training new CatBoost model...")
    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42,
        train_dir="models/catboost_logs",
        class_weights=[1, scale_pos_weight]
    )
    cat_model.fit(X_train, y_train)
    cat_model.save_model("models/cat_model.cbm")
    print("New CatBoost model trained and saved!")

# Predictions
y_pred_cat = cat_model.predict(X_test)
y_proba_cat = cat_model.predict_proba(X_test)[:, 1]

# Evaluation
print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_cat))
print("ROC AUC (CatBoost):", roc_auc_score(y_test, y_proba_cat))
"""
09_visualization.py
-------------------
Visualizations for model performance and feature importance.
- ROC Curve Comparison
- Precision-Recall Curve Comparison
- Top 10 Feature Importances for XGBoost and CatBoost
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score, roc_auc_score
from xgboost import XGBClassifier, plot_importance
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

# ROC Curve Comparison
fig, ax = plt.subplots(figsize=(7,6))
RocCurveDisplay.from_predictions(y_test, y_proba_xgb, name="XGBoost", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_cat, name="CatBoost", ax=ax)
ax.set_title("ROC Curve Comparison", fontsize=14, weight="bold")
ax.grid(linestyle="--", alpha=0.6)
plt.show()

# Precision-Recall Curve Comparison
fig, ax = plt.subplots(figsize=(7,6))
PrecisionRecallDisplay.from_predictions(
    y_test, y_proba_xgb, name=f"XGBoost (AP={average_precision_score(y_test, y_proba_xgb):.2f})", ax=ax
)
PrecisionRecallDisplay.from_predictions(
    y_test, y_proba_cat, name=f"CatBoost (AP={average_precision_score(y_test, y_proba_cat):.2f})", ax=ax
)
ax.set_title("Precision-Recall Curve Comparison", fontsize=14, weight="bold")
ax.grid(linestyle="--", alpha=0.6)
plt.show()

# XGBoost Feature Importance
plot_importance(xgb_model, max_num_features=10)
plt.gcf().set_size_inches(10, 6)
plt.title("XGBoost - Top 10 Feature Importances", fontsize=14, weight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# CatBoost Feature Importance
cat_importances = cat_model.get_feature_importance()
cat_features = X.columns

fi_df = pd.DataFrame({"Feature": cat_features, "Importance": cat_importances})
fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)

sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", legend=False)
plt.gcf().set_size_inches(10, 6)
plt.title("CatBoost - Top 10 Feature Importances", fontsize=14, weight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
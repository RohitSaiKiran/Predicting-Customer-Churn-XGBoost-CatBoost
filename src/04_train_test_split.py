"""
04_train_test_split.py
----------------------
Split the preprocessed dataset into training and testing sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("Data/processed_churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)
"""
run_all.py
----------
Main runner script for the Churn Prediction project.

Executes all steps: imports, preprocessing, EDA, splitting, training, evaluation, and demo.

=============================
Project Configuration Guide
=============================
Here are the main ways you can experiment with the project:

1. Demo toggle (this file):
   - Set `include_demo = True` to run the demo (src/10_demo.py).
   - Set `include_demo = False` to skip the demo.

2. Model training flags (inside src/05_xgboost.py and src/06_catboost.py):
   - `use_pretrained_xgb = True/False` → load or train XGBoost model  
   - `use_pretrained_cat = True/False` → load or train CatBoost model

3. Live demo customer index (inside src/10_demo.py):
   - Change `sample = X_test.iloc[0]` to another index.
   - Example: `sample = X_test.iloc[42]` → predicts for the 43rd customer.
   - ⚠️ Valid indices: 0 to 1408 (1,409 test customers).
     Using a number outside this range will cause an error.

Reminder: If you tweak any of these settings, don’t forget to SAVE the file.
"""

import subprocess

# Toggle whether to run the demo at the end
include_demo = True

# Ordered pipeline modules
modules = [
    "src/01_imports.py",
    "src/02_data_preprocessing.py",
    "src/03_eda.py",
    "src/04_train_test_split.py",
    "src/05_xgboost.py",
    "src/06_catboost.py",
    "src/07_results.py",
    "src/08_visualization.py",
]

# Add demo only if enabled
if include_demo:
    modules.append("src/09_demo.py")

# Run pipeline
for module in modules:
    print(f"\n===== Running {module} =====")
    subprocess.run(["python3", module], check=True)

print("\n✅ Pipeline completed successfully!")
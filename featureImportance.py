# file: make_feature_importance.py
# Computes permutation feature importance on the hold-out set and saves CSV + plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from pycaret.classification import load_model
from pathlib import Path

# === paths ===
BASE          = Path(__file__).resolve().parent
TRAIN_ROOT    = BASE / "Outputs" / "TrainModel"
MODEL_BASENAME = TRAIN_ROOT / "models" / "best_heart_model"     # suffix .pkl will be added by load_model() So we dont declare suffix here
PREDS_FILE     = TRAIN_ROOT / "metrics" / "preds_holdout.csv"

OUT_DIR      = BASE / "Outputs" / "FeatureImportance"
METRICS_DIR  = OUT_DIR / "metrics"
PLOTS_DIR    = OUT_DIR / "plots"
for d in (METRICS_DIR, PLOTS_DIR): d.mkdir(parents=True, exist_ok=True)

FI_CSV = METRICS_DIR / "feature_importance_permutation.csv"
FI_PNG = PLOTS_DIR / "Feature_Importance_permutation.png"

# 1) load finalized pipeline from best_heart_model.pkl
best = load_model(str(MODEL_BASENAME))

# 2) use preds_holdout.csv for original features & labels
#    (Run Trainmodel.py first to create this.)
df = pd.read_csv(PREDS_FILE)

target_col = next(c for c in ["Label","label","Target","target"] if c in df.columns)
drop_cols = {target_col, "prediction_label", "prediction_score", "Score", "probability"}

X = df[[c for c in df.columns if c not in drop_cols]]
y = df[target_col].astype(int).to_numpy()

# 3) permutation importance (F1 is a good choice for medical screening)
res = permutation_importance(best, X, y, n_repeats=10, random_state=42, scoring="f1")
imp = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": res.importances_mean,
    "importance_std":  res.importances_std
}).sort_values("importance_mean", ascending=True)  # ascending -> nicer horizontal bars

imp.to_csv(FI_CSV, index=False)

# 4) plot the feature importance
plt.figure(figsize=(8, max(4, len(imp)*0.3)))
plt.barh(imp["feature"], imp["importance_mean"])
plt.xlabel("Permutation importance (mean ΔF1)")
plt.title("Feature Importance (Permutation on Hold-out)")
plt.tight_layout()
plt.savefig(FI_PNG, dpi=200)
plt.show()

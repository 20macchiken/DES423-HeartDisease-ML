# file: make_feature_importance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from pycaret.classification import load_model

# 1) load finalized pipeline (no .pkl suffix)
best = load_model("best_heart_model")

# 2) use preds_holdout.csv to get the original features & labels
df = pd.read_csv("preds_holdout.csv")

target_col = next(c for c in ["Label","label","Target","target"] if c in df.columns)
drop_cols = {target_col, "prediction_label", "prediction_score", "Score", "probability"}

X = df[[c for c in df.columns if c not in drop_cols]]
y = df[target_col].astype(int).to_numpy()

# 3) permutation importance (F1 by default a good choice too; here we'll use "f1")
res = permutation_importance(best, X, y, n_repeats=10, random_state=42, scoring="f1")
imp = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": res.importances_mean,
    "importance_std":  res.importances_std
}).sort_values("importance_mean", ascending=True)  # ascending for a nice horizontal bar

imp.to_csv("feature_importance_permutation.csv", index=False)

# 4) plot
plt.figure(figsize=(8, max(4, len(imp)*0.3)))
plt.barh(imp["feature"], imp["importance_mean"])
plt.xlabel("Permutation importance (mean ΔF1)")
plt.title("Feature Importance (Permutation on Hold-out)")
plt.tight_layout()
plt.savefig("Feature_Importance_permutation.png", dpi=200)
plt.show()

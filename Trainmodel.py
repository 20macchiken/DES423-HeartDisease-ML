# Trainmodel.py
# Trains & compares multiple models with PyCaret (classification),
# tunes the best, evaluates on hold-out, and saves metrics + model.

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pycaret.classification import (
    setup, compare_models, tune_model, predict_model, finalize_model, save_model, pull
)

# ====== CONFIG ======
CSV_PATH = "data_clean/heart_all_clean_binary.csv"  # adjust if needed
TARGET   = "target"
SESSION  = 42
SORT_MET = "F1"   # F1 is good for medical screening

# ====== LOAD DATA ======
df = pd.read_csv(CSV_PATH)
# Drop dataset identifier to avoid shortcut learning
if "source" in df.columns:
    df = df.drop(columns=["source"])

# ====== SETUP (split + CV + preprocessing) ======
setup(
    data=df,
    target=TARGET,
    session_id=SESSION,
    train_size=0.8,
    fold=5,
    fold_strategy="stratifiedkfold",
    normalize=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    log_experiment=False,
    verbose=False,
)

# ====== 1) COMPARE MULTIPLE MODELS ======
top_models = compare_models(n_select=5, sort=SORT_MET)
lb_compare = pull().copy()  # leaderboard after compare
lb_compare.to_csv("leaderboard_compare.csv", index=False)

# ====== 2) TUNE THE BEST MODEL ======
best_base = top_models[0] if isinstance(top_models, list) else top_models
best_tuned = tune_model(best_base, optimize=SORT_MET, choose_better=True)
lb_tune = pull().copy()
lb_tune.to_csv("leaderboard_tune.csv", index=False)

# ====== 3) EVALUATE ON HOLD-OUT ======
preds = predict_model(best_tuned)   # returns a DataFrame with predictions
metrics_tbl = pull().copy()         # table shown by PyCaret after predict_model

# ---- robust metric extraction (works for wide/long tables or computes directly) ----
def _extract_from_table(tbl, key_names):
    """Try to read metric from either wide (one-row) or long ['Metric','Value'] tables."""
    if tbl is None or tbl.empty:
        return None
    # Wide table (columns like 'Accuracy','Precision','Recall','F1','AUC')
    for k in key_names:
        if k in tbl.columns:
            try:
                return float(tbl.iloc[0][k])
            except Exception:
                pass
    # Long table (columns 'Metric','Value')
    if "Metric" in tbl.columns and "Value" in tbl.columns:
        for k in key_names:
            row = tbl[tbl["Metric"].astype(str).str.lower() == k.lower()]
            if not row.empty:
                try:
                    return float(row["Value"].values[0])
                except Exception:
                    pass
    return None

acc  = _extract_from_table(metrics_tbl, ["Accuracy"])
prec = _extract_from_table(metrics_tbl, ["Precision", "Prec."])
rec  = _extract_from_table(metrics_tbl, ["Recall"])
f1   = _extract_from_table(metrics_tbl, ["F1", "F1-Score"])
auc  = _extract_from_table(metrics_tbl, ["AUC", "ROC AUC"])

# If anything missing, compute from preds
y_true = None
y_pred = None
y_score = None
for cand in ["Label", "label", "Target", "target"]:
    if cand in preds.columns:
        y_true = preds[cand].to_numpy()
        break
for cand in ["prediction_label", "Predicted Label", "prediction"]:
    if cand in preds.columns:
        y_pred = preds[cand].to_numpy()
        break
for cand in ["prediction_score", "Score", "probability"]:
    if cand in preds.columns:
        y_score = preds[cand].to_numpy()
        break

if y_true is not None and y_pred is not None:
    if acc is None:  acc  = float(accuracy_score(y_true, y_pred))
    if prec is None: prec = float(precision_score(y_true, y_pred, zero_division=0))
    if rec is None:  rec  = float(recall_score(y_true, y_pred, zero_division=0))
    if f1 is None:   f1   = float(f1_score(y_true, y_pred, zero_division=0))
    if auc is None and y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass

# Save a clean CSV with the required metrics
metrics_out = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "AUC": auc
}])
metrics_out.to_csv("metrics_holdout.csv", index=False)

print("\n=== SUMMARY (Hold-out) ===")
print("Best model (tuned):", best_tuned)
print(
    f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  "
    f"F1={f1:.4f}  AUC={(auc if auc is not None else float('nan')):.4f}"
)

# ====== 4) FINALIZE & SAVE MODEL ======
final_model = finalize_model(best_tuned)       # retrain on full data with tuned params
save_model(final_model, "best_heart_model")    # -> best_heart_model.pkl

print("\nSaved files:")
print(" - leaderboard_compare.csv")
print(" - leaderboard_tune.csv")
print(" - metrics_holdout.csv")
print(" - best_heart_model.pkl")

# make_confusion.py
# Prints Accuracy + sklearn classification_report (pretty table)
# and saves the confusion-matrix plot.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# === paths ===
BASE         = Path(__file__).resolve().parent
TRAIN_METRICS = BASE / "Outputs" / "TrainModel" / "metrics"         # use this path for data that were produced by Trainmodel.py
OUT_DIR       = BASE / "Outputs" / "ConfMatrix"
METRICS_DIR   = OUT_DIR / "metrics"
PLOTS_DIR     = OUT_DIR / "plots"
for d in (METRICS_DIR, PLOTS_DIR): d.mkdir(parents=True, exist_ok=True)

PREDS_FILE    = TRAIN_METRICS / "preds_holdout.csv"                  # input
REPORT_TXT    = METRICS_DIR / "classification_report.txt"            # outputs
CM_PNG        = PLOTS_DIR / "Confusion_Matrix_manual.png"

# === config (optional: rename labels for display) ===
# For heart-disease binary: 0 = no_disease, 1 = disease

CLASS_LABELS = {0: "no_disease", 1: "disease"}   # e.g. soil: {0:"bulk", 1:"rhizosphere"}

# 1) load predictions saved earlier by Trainmodel.py
df = pd.read_csv(PREDS_FILE)

# 2) infer column names
y_col  = next(c for c in ["Label","label","Target","target"] if c in df.columns)
yh_col = next(c for c in ["prediction_label","Predicted Label","prediction"] if c in df.columns)

y_true = df[y_col].astype(int).to_numpy()
y_pred = df[yh_col].astype(int).to_numpy()

# 3) accuracy + classification report (pretty)
acc = accuracy_score(y_true, y_pred)

# build target_names in ascending class order
unique_sorted = np.sort(np.unique(y_true))
target_names  = [CLASS_LABELS.get(int(k), str(k)) for k in unique_sorted]

print(f"\nAccuracy: {acc*100:,.2f}%\n")
print(classification_report(y_true, y_pred, target_names=target_names, digits=2))

# raw confusion matrix printed in terminal
conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_sorted)
print(conf_matrix)

# save text report
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc*100:,.2f}%\n\n")
    f.write(classification_report(y_true, y_pred, target_names=target_names, digits=2))
    f.write("\n" + np.array2string(conf_matrix))

# 4) plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)
disp.plot(values_format="d")
plt.title("Confusion Matrix (Hold-out)")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=200)
plt.show()

# make_confusion.py
# Prints Accuracy + sklearn classification_report (pretty table)
# and saves the same confusion-matrix plot you already had.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# === config (optional: rename labels for display) ===
# For heart-disease binary: 0 = no_disease, 1 = disease
CLASS_LABELS = {0: "no_disease", 1: "disease"}   # change to {0:"bulk",1:"rhizosphere"} for soil task
# ====================================================

# 1) load predictions saved earlier
df = pd.read_csv("preds_holdout.csv")

# 2) infer column names
y_col  = next(c for c in ["Label","label","Target","target"] if c in df.columns)
yh_col = next(c for c in ["prediction_label","Predicted Label","prediction"] if c in df.columns)

y_true = df[y_col].astype(int).to_numpy()
y_pred = df[yh_col].astype(int).to_numpy()

# 3) accuracy + classification report (pretty)
acc = accuracy_score(y_true, y_pred)

# build target_names in correct order [0,1] if possible
unique_sorted = np.sort(np.unique(y_true))
target_names = [CLASS_LABELS.get(int(k), str(k)) for k in unique_sorted]

print(f"\nAccuracy: {acc*100:,.2f}%\n")
print(classification_report(y_true, y_pred, target_names=target_names, digits=2))

# raw confusion matrix array printed in the console / terminal
conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_sorted)
print(conf_matrix)

# Classification report + confusion matrix saved to a text file

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc*100:,.2f}%\n\n")
    f.write(classification_report(y_true, y_pred, target_names=target_names, digits=2))
    f.write("\n" + np.array2string(conf_matrix))

# 4) plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)
disp.plot(values_format="d")
plt.title("Confusion Matrix (Hold-out)")
plt.tight_layout()
plt.savefig("Confusion_Matrix_manual.png", dpi=200)
plt.show()

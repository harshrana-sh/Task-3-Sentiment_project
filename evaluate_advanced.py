"""
evaluate_advanced.py
---------------------
Evaluates the lexicon-based "advanced model" substitute (see
lexicon_model.py for why this stands in for a pretrained transformer) on
the SAME held-out test split used for the baseline, then appends it to
outputs/metrics_table.csv and regenerates the comparison chart, plus a
dedicated confusion matrix image for this model.
"""
import sys
sys.path.insert(0, "/home/claude/sentiment_project/scripts")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

from lexicon_model import predict_batch

DATA_PATH = "/home/claude/sentiment_project/data/reviews.csv"
OUTPUTS_DIR = "/home/claude/sentiment_project/outputs"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
# Use the identical split as train_baseline.py (same random_state/stratify/test_size)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)

preds = predict_batch(X_test.tolist())

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
prec = precision_score(y_test, preds, average="macro")
rec = recall_score(y_test, preds, average="macro")
print(f"Lexicon (advanced substitute) model -> accuracy={acc:.3f} f1_macro={f1:.3f}")

# Append to metrics table
metrics_df = pd.read_csv(f"{OUTPUTS_DIR}/metrics_table.csv")
metrics_df = metrics_df[metrics_df["model"] != "Lexicon Rule-Based (advanced substitute)"]
new_row = pd.DataFrame([{
    "model": "Lexicon Rule-Based (advanced substitute)",
    "accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1
}])
metrics_df = pd.concat([metrics_df, new_row], ignore_index=True).sort_values("f1_macro", ascending=False)
metrics_df.to_csv(f"{OUTPUTS_DIR}/metrics_table.csv", index=False)
print(metrics_df)

# Confusion matrix for the lexicon model
labels_order = sorted(df["label"].unique())
cm = confusion_matrix(y_test, preds, labels=labels_order)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Lexicon Rule-Based (advanced substitute)")
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/confusion_matrix_advanced.png", dpi=150)
plt.close()

# Updated comparison chart across all models
plt.figure(figsize=(8, 4.5))
melted = metrics_df.melt(id_vars="model", value_vars=["accuracy", "f1_macro"],
                          var_name="metric", value_name="score")
sns.barplot(data=melted, x="model", y="score", hue="metric")
plt.ylim(0, 1)
plt.title("Model Comparison (Baseline vs Advanced Substitute)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/model_comparison.png", dpi=150)
plt.close()

print(f"\nSaved updated metrics table and plots to {OUTPUTS_DIR}/")

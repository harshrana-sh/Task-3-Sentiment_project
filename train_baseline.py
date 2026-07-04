"""
train_baseline.py
------------------
Baseline + comparison models for sentiment classification:
  1. TF-IDF + Logistic Regression   (the required "baseline")
  2. TF-IDF + Linear SVM            (comparison)
  3. TF-IDF + Multinomial Naive Bayes (comparison)

Outputs:
  - models/tfidf_vectorizer.joblib
  - models/logreg_model.joblib        (best/primary saved model)
  - outputs/metrics_table.csv
  - outputs/confusion_matrix.png      (for the primary Logistic Regression model)
  - outputs/model_comparison.png
"""
import sys
sys.path.insert(0, "/home/claude/sentiment_project/scripts")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from preprocess import preprocess_for_tfidf

DATA_PATH = "/home/claude/sentiment_project/data/reviews.csv"
MODELS_DIR = "/home/claude/sentiment_project/models"
OUTPUTS_DIR = "/home/claude/sentiment_project/outputs"

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(preprocess_for_tfidf)
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows | class balance:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        results.append({"model": name, "accuracy": acc, "precision_macro": prec,
                         "recall_macro": rec, "f1_macro": f1})
        trained[name] = (model, preds)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds))

    metrics_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(f"{OUTPUTS_DIR}/metrics_table.csv", index=False)
    print("\nMetrics table:\n", metrics_df)

    # Primary / saved model = Logistic Regression (the specified baseline)
    best_model, best_preds = trained["Logistic Regression"]

    # Confusion matrix for the primary model
    labels_order = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels_order)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — TF-IDF + Logistic Regression")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(7, 4.5))
    melted = metrics_df.melt(id_vars="model", value_vars=["accuracy", "f1_macro"],
                              var_name="metric", value_name="score")
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # Save vectorizer + primary model
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(best_model, f"{MODELS_DIR}/logreg_model.joblib")
    print(f"\nSaved vectorizer and Logistic Regression model to {MODELS_DIR}/")
    print(f"Saved metrics table and plots to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()

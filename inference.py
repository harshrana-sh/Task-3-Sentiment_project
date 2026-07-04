"""
inference.py
-------------
Reusable inference script: loads the saved TF-IDF vectorizer + Logistic
Regression model and predicts sentiment for new text.

Usage:
    python3 inference.py "This laptop is amazing, best purchase ever!"
    python3 inference.py --file my_reviews.txt   # one review per line
    python3 inference.py                          # interactive mode
"""
import sys
import argparse
import joblib

sys.path.insert(0, "/home/claude/sentiment_project/scripts")
from preprocess import preprocess_for_tfidf

MODELS_DIR = "/home/claude/sentiment_project/models"


def load_model():
    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    model = joblib.load(f"{MODELS_DIR}/logreg_model.joblib")
    return vectorizer, model


def predict(texts, vectorizer, model):
    cleaned = [preprocess_for_tfidf(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    results = []
    for t, p, pr in zip(texts, preds, probs):
        confidence = pr[classes.index(p)]
        results.append((t, p, confidence))
    return results


def main():
    parser = argparse.ArgumentParser(description="Sentiment inference (TF-IDF + Logistic Regression)")
    parser.add_argument("text", nargs="?", help="A single review to classify")
    parser.add_argument("--file", help="Path to a text file, one review per line")
    args = parser.parse_args()

    vectorizer, model = load_model()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        print("Enter a review (blank line to quit):")
        texts = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            texts.append(line)

    if not texts:
        print("No input provided.")
        return

    results = predict(texts, vectorizer, model)
    for text, label, confidence in results:
        print(f"[{label.upper():>8} | {confidence:.1%}]  {text}")


if __name__ == "__main__":
    main()

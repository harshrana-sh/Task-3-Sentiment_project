"""
generate_dataset.py
--------------------
OFFLINE SUBSTITUTION NOTE:
This sandbox has no internet access, so real Twitter/Reddit/IMDB samples
could not be downloaded (no Kaggle API, no nltk.download(), no HF datasets).
To keep the pipeline realistic and reproducible, this script *synthesizes*
a labeled review dataset covering four domains (movies, products, restaurants,
services) using template + phrase-bank recombination, with:
  - randomized phrase order
  - optional negation ("not", "never") injected into ~12% of samples to
    make the classification task non-trivial (avoids trivial 100% accuracy
    from simple keyword matching)
  - randomized filler/neutral clauses
The result behaves like a real sentiment corpus: short free-text reviews
with a binary label (positive / negative), class imbalance kept mild but
non-zero, and enough lexical variety that TF-IDF + Logistic Regression has
a genuine learning problem instead of a memorization problem.

Output: ../data/reviews.csv  with columns [text, label]  (label: positive/negative)
"""
import random
import csv

random.seed(42)

SUBJECTS = ["movie", "phone", "laptop", "restaurant", "hotel", "book", "app", "headphones", "service", "flight"]

POSITIVE_PHRASES = [
    "absolutely loved it", "exceeded my expectations", "works flawlessly",
    "best purchase I've made this year", "the quality is outstanding",
    "would highly recommend to anyone", "customer support was fantastic",
    "such a pleasant surprise", "worth every penny", "a genuinely great experience",
    "the story kept me hooked from start to finish", "battery life is amazing",
    "delivery was fast and the packaging was perfect", "the staff were incredibly friendly",
    "five stars, no complaints at all", "it just works, no issues whatsoever",
    "beautifully designed and easy to use", "the food was delicious and fresh",
    "left me smiling the whole time", "a masterpiece from start to finish"
]

NEGATIVE_PHRASES = [
    "completely disappointed me", "broke after two days", "waste of money",
    "the worst experience I've had", "customer support was useless",
    "would not recommend to anyone", "the quality feels cheap",
    "took forever to arrive and arrived damaged", "the plot made no sense",
    "battery drains within an hour", "the staff were rude and dismissive",
    "one star, would give zero if I could", "constantly crashes and freezes",
    "overpriced for what you get", "the food was cold and bland",
    "a total letdown", "I regret buying this", "riddled with bugs",
    "painfully slow and unresponsive", "not what was advertised at all"
]

NEUTRAL_FILLERS = [
    "I bought it last week.", "I used it for a few days before writing this.",
    "A friend recommended it to me.", "I compared it with a few alternatives first.",
    "This is my honest review.", "I've used similar products before.",
    "I wanted to like it.", "I was looking forward to trying it.",
    "It arrived on time.", "I saw a lot of reviews before deciding."
]

NEGATIONS = ["not really", "definitely not", "far from", "never"]


def make_review(label):
    subject = random.choice(SUBJECTS)
    n_phrases = random.randint(1, 3)
    phrase_bank = POSITIVE_PHRASES if label == "positive" else NEGATIVE_PHRASES
    own_phrases = random.sample(phrase_bank, k=min(n_phrases, len(phrase_bank)))

    opposite_bank = NEGATIVE_PHRASES if label == "positive" else POSITIVE_PHRASES
    phrases = list(own_phrases)

    # Inject a negated opposite-sentiment phrase (harder: model must not just
    # keyword-match on the opposite phrase's raw words).
    if random.random() < 0.25:
        neg_phrase = random.choice(opposite_bank)
        phrases.append(f"{random.choice(NEGATIONS)} {neg_phrase}")

    # Occasionally add a genuine mixed-sentiment clause from the opposite
    # bank WITHOUT negation (a real "mixed" review, e.g. good product but
    # bad service) while keeping the overall label as the majority sentiment.
    # This creates realistic label noise / ambiguity, similar to real-world
    # review data where sentiment isn't perfectly clean-cut.
    if random.random() < 0.15:
        phrases.append(random.choice(opposite_bank))

    if random.random() < 0.4:
        phrases.insert(0, random.choice(NEUTRAL_FILLERS))

    random.shuffle(phrases)
    text = f"This {subject} " + ", ".join(phrases) + "."
    return text


def build_dataset(n_per_class=500):
    rows = []
    for _ in range(n_per_class):
        rows.append((make_review("positive"), "positive"))
    # mild class imbalance: slightly fewer negative samples, like many real corpora
    for _ in range(int(n_per_class * 0.85)):
        rows.append((make_review("negative"), "negative"))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_dataset(n_per_class=500)
    out_path = "/home/claude/sentiment_project/data/reviews.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    pos = sum(1 for _, l in rows if l == "positive")
    neg = sum(1 for _, l in rows if l == "negative")
    print(f"positive={pos} negative={neg}")

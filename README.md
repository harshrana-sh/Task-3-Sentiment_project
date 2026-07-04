# Task 3 — Sentiment Analysis

TF-IDF + Logistic Regression baseline, with Linear SVM and Multinomial Naive
Bayes comparisons, plus a lexicon-based rule scorer standing in for a
pretrained transformer ("advanced" tier). Full pipeline: dataset → cleaning
→ training → evaluation → saved model → CLI inference.

## ⚠️ Offline sandbox substitutions

This project was built in a sandbox with **no internet access**. Two parts
of the original spec required substitutions, both documented in code comments
at the point they occur:

| Spec asked for | Not available because | Substitute used |
|---|---|---|
| Twitter/Reddit/IMDB dataset | No Kaggle API, no `nltk.download()`, no HF `datasets` | Synthetic labeled review dataset generated in `scripts/generate_dataset.py`, engineered with negation and mixed-sentiment noise so the task isn't trivially easy |
| `transformers`/HF pretrained model ("Advanced" tier) | `pip install transformers` fails (no PyPI mirror); no weights downloadable | Lexicon + negation-aware rule-based scorer (`scripts/lexicon_model.py`), evaluated *without* training on this dataset's labels — same role as an off-the-shelf model |
| `nltk` tokenizer/stopwords | `nltk` not installed, cannot be downloaded | Custom regex tokenizer + hand-compiled stopword list, `scripts/preprocess.py` |

If you have network access, swapping in the real components is a small
change — see the comments at the top of `lexicon_model.py` for the two-line
Hugging Face `pipeline()` equivalent, and `generate_dataset.py`'s docstring
for where to plug in a real IMDB/Kaggle CSV instead (just point
`train_baseline.py`'s `DATA_PATH` at your file with `text`/`label` columns).

## Project structure

```
sentiment_project/
├── data/
│   └── reviews.csv                  # generated dataset (text, label)
├── scripts/
│   ├── generate_dataset.py          # synthetic dataset generator
│   ├── preprocess.py                # cleaning / tokenization (no nltk)
│   ├── train_baseline.py            # TF-IDF + LogReg/SVM/NB training + eval
│   ├── lexicon_model.py             # advanced-tier substitute
│   ├── evaluate_advanced.py         # evaluates lexicon model, updates outputs
│   └── inference.py                 # CLI inference on the saved model
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── logreg_model.joblib
├── outputs/
│   ├── metrics_table.csv
│   ├── confusion_matrix.png         # Logistic Regression (primary model)
│   ├── confusion_matrix_advanced.png
│   └── model_comparison.png
├── notebooks/
│   └── sentiment_analysis.ipynb     # end-to-end training notebook
└── README.md
```

## How to run

```bash
cd scripts
python3 generate_dataset.py      # writes ../data/reviews.csv
python3 train_baseline.py        # trains + saves models, writes metrics/plots
python3 evaluate_advanced.py     # evaluates lexicon "advanced" model
python3 inference.py "This product is amazing!"
```

## Results summary

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Lexicon Rule-Based (advanced substitute) | 0.924 | 0.923 |
| Logistic Regression (**saved/primary**) | 0.908 | 0.907 |
| Multinomial Naive Bayes | 0.903 | 0.901 |
| Linear SVM | 0.886 | 0.885 |

The lexicon model edges out the trained models here largely because its
hand-built vocabulary overlaps with the synthetic phrase bank used to
generate the dataset — on genuinely novel review text, a corpus-trained
model (or a real pretrained transformer) would be expected to generalize
better. Logistic Regression is the strongest of the trained models and is
the one saved to `models/` for reuse.

## Learning outcomes covered

- **Text cleaning & tokenization**: regex-based cleaning, custom stopword
  removal (`preprocess.py`)
- **Model inference**: TF-IDF vectorization + classifier `.predict()`,
  wrapped in a reusable CLI (`inference.py`)
- **Evaluation metrics**: accuracy, precision/recall/F1 (macro), confusion
  matrices, model comparison chart

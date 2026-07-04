"""
preprocess.py
-------------
OFFLINE SUBSTITUTION NOTE: nltk is not installed and cannot be downloaded in
this sandbox (no network access, no PyPI mirror, nltk.download() would also
require network for corpora like 'punkt' and 'stopwords'). This module
reimplements the two nltk features this project needs, using only the
standard library:
  - tokenize(): simple regex-based word tokenizer (lowercase, strips
    punctuation/numbers), equivalent in effect to nltk.word_tokenize()
    for this use case.
  - STOPWORDS: a hand-compiled list of ~120 common English stopwords,
    equivalent in effect to nltk.corpus.stopwords.words('english').
"""
import re

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    text = clean_text(text)
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Return a cleaned, stopword-free string ready for TfidfVectorizer."""
    return " ".join(tokenize(text))


if __name__ == "__main__":
    sample = "This movie ABSOLUTELY loved it! Best purchase I've made this year, 10/10."
    print("Original :", sample)
    print("Cleaned  :", preprocess_for_tfidf(sample))

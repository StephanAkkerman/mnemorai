from collections import Counter
from itertools import chain

from nltk import bigrams, download
from nltk.corpus import brown, gutenberg, reuters, webtext
from wordfreq import zipf_frequency

# Download the necessary NLTK corpora if not already present
for c in ["reuters", "gutenberg", "webtext"]:
    download(c, quiet=True)

# Zipf → raw unigram probability.  Zipf 6 ≈ 1/1 000, so P = 10**(zipf−9)
_unigram_p = lambda w: 10 ** (zipf_frequency(w, "en") - 9)

tokens = [
    w.lower()
    for w in chain(brown.words(), reuters.words(), gutenberg.words(), webtext.words())
]
_unigram_cnt = Counter(tokens)
_bigram_cnt = Counter(bigrams(tokens))
_V = len(_unigram_cnt)


def backoff_prob(w1: str, w2: str) -> float:
    """Naïve product P(w1)·P(w2) from the Wordfreq unigram model."""
    return _unigram_p(w1) * _unigram_p(w2)


def brown_bigram_prob(w1: str, w2: str) -> float:
    # Laplace-smoothed P(w2 | w1)
    return (_bigram_cnt[(w1, w2)] + 1) / (_unigram_cnt[w1] + _V)


def bigram_prob(w1: str, w2: str, *, alpha: float = 0.1) -> float:
    """
    Hybrid probability.

      • if Brown corpus has seen (w1,w2), return its Laplace value
      • otherwise fall back to alpha·P_wordfreq(w1)·P_wordfreq(w2)

    `alpha` (default 0.1) keeps back-off numbers on the same scale as
    real bigram counts—tune it if the gap feels too big or small.
    """
    base = brown_bigram_prob(w1, w2)
    if _bigram_cnt[(w1, w2)] == 0:  # unseen → use back-off
        return alpha * backoff_prob(w1, w2)
    return base


def bigram_grid(list1, list2, *, sort_desc: bool = True):
    """Return (w1, w2, hybrid-prob) for every w1∈list1, w2∈list2."""
    out = [
        (w1, w2, bigram_prob(w1.lower(), w2.lower()))  # ← CHANGED
        for w1 in list1
        for w2 in list2
    ]
    return sorted(out, key=lambda t: t[2], reverse=sort_desc) if sort_desc else out


if __name__ == "__main__":
    unformatted_list1 = (
        "duh, ta, tea, toe, tie, dew, due, doe, dough, die, dart, door, "
        "thaw, though, there, tire, dare, donor, draw, tear, door, data, "
        "deter, tune, the"
    )
    unformatted_list2 = (
        "sing, sink, sting, thing, wing, king, cling, grin, gin, gang, gone, "
        "gong, then, thens, ten, tang, tan, town, tongue, tinge, begin, bing, "
        "singe, swing, twin"
    )
    list1 = unformatted_list1.split(", ")
    list2 = unformatted_list2.split(", ")

    for w1, w2, p in bigram_grid(list1, list2):
        print(f"{w1} {w2:<10}  P={p:.3e}")

# Maybe also do levenshtein distance between for the chunks

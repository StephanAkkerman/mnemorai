from collections import Counter
from itertools import chain, product

import pandas as pd
from nltk import download
from nltk.corpus import brown, gutenberg, reuters, webtext
from nltk.util import ngrams
from wordfreq import zipf_frequency

# ――― downloads + token loading ―――
for c in ["reuters", "gutenberg", "webtext"]:
    download(c, quiet=True)

tokens = [
    w.lower()
    for w in chain(brown.words(), reuters.words(), gutenberg.words(), webtext.words())
]

# ――― unigram P from zipf → back-off base ―――
_unigram_p = lambda w: 10 ** (zipf_frequency(w, "en") - 9)

# ――― build counters for all orders up to MAX_N ―――
MAX_N = 5  # pick the highest n-gram you want
counters = {
    1: Counter(tokens),
    **{n: Counter(ngrams(tokens, n)) for n in range(2, MAX_N + 1)},
}

V = len(counters[1])  # vocab size


def ngram_prob(gram: tuple[str, ...], *, alpha: float = 0.1) -> float:
    """
    If seen: Laplace-smoothed P(w_n | w1…w_{n-1}).

    Else: back off to P(w_n | w2…w_{n-1}) x alpha, recursively until unigram.
    """
    n = len(gram)
    if n == 1:
        return _unigram_p(gram[0])

    cnt_ng = counters[n][gram]
    cnt_ctx = counters[n - 1][gram[:-1]]

    if cnt_ng > 0:
        # (C(gram) + 1) / (C(context) + V)
        return (cnt_ng + 1) / (cnt_ctx + V)
    else:
        # back-off: drop the first word in context
        return alpha * ngram_prob(gram[1:], alpha=alpha)


def ngram_grid(lists: list[list[str]], *, alpha: float = 0.1, sort_desc: bool = True):
    """
    lists: a list of N word-lists → builds all N-grams.

    returns: [(tuple_of_words, P), …], sorted by P if desired.
    """
    N = len(lists)
    if N < 1 or N > MAX_N:
        raise ValueError(f"N must be 1-{MAX_N}, got {N}")

    out = []
    for combo in product(*lists):
        gram = tuple(w.lower() for w in combo)
        p = ngram_prob(gram, alpha=alpha)
        out.append((gram, p))

    return sorted(out, key=lambda x: x[1], reverse=sort_desc) if sort_desc else out


def ngrams_to_df(
    *ngram_specs: list[list[str]], alpha: float = 0.1, sort_desc: bool = True
) -> pd.DataFrame:
    """
    Build a DataFrame of n-gram probabilities for one or more n-gram specs.

    Parameters
    ----------
    *ngram_specs : list of list of str
        Each spec is itself a list of word-lists, e.g.
        unigram:   [ ["a", "b", "c"] ]
        bigram:    [ ["the", "a"], ["cat", "dog"] ]
        trigram:   [ ["I", "You"], ["saw", "like"], ["it", "them"] ]
        You can pass multiple specs to get them all in one DataFrame.
    alpha : float, optional
        back-off weight (passed to `ngram_prob`), by default 0.1
    sort_desc : bool, optional
        whether to sort by descending probability, by default True

    Returns
    -------
    pd.DataFrame
        columns = ["ngram", "prob"], where “ngram” is the words joined by spaces.
    """
    dfs = []
    for spec in ngram_specs:
        # run our existing ngram_grid
        results = ngram_grid(spec, alpha=alpha, sort_desc=sort_desc)
        # turn into rows of (joined-string, prob)
        rows = [(" ".join(gram), p) for gram, p in results]
        df = pd.DataFrame(rows, columns=["ngram", "prob"])
        dfs.append(df)
    # concatenate all specs into one table
    new_df = pd.concat(dfs, ignore_index=True)
    # sort by probability if requested
    if sort_desc:
        new_df = new_df.sort_values(by="prob", ascending=False).reset_index(drop=True)
    return new_df


# ――― example usage ―――
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

    input_dict = {
        "da": unformatted_list1.split(", "),
        "ging": unformatted_list2.split(", "),
    }

    print(ngrams_to_df([input_dict["da"]], [input_dict["da"], input_dict["ging"]]))

    # Unigrams
    # for (w1), p in ngram_grid([da]):
    #     print(f"{w1} P={p:.3e}")

    # # Bigrams
    # for (w1, w2), p in ngram_grid([da, ging]):
    #     print(f"{w1} {w2:<10} P={p:.3e}")

# Maybe also do levenshtein distance between for the chunks

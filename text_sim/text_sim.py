# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from collections import Counter
from datetime import timedelta
from functools import update_wrapper, partial
import re

from gensim import corpora, models, similarities

from .stopwords import stopwords

stopwords_re = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
non_an = re.compile(r'\W')
collapse = re.compile(r'\s')


class memoize_method(object):
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:])
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args)
        return res

def memoize(f):
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)
        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
    return memodict().__getitem__


@memoize
def clean_text(s):
    s = s.lower().strip()
    s = stopwords_re.sub('', s)
    s = non_an.sub(' ', s)
    s = collapse.sub(' ', s)
    return s


@memoize
def make_ngrams(s, ngrams=2):
    return [s[i:i+ngrams] for i in range(len(s)-(ngrams-1))]

@memoize
def tokenize(s, ngrams):
    return make_ngrams(clean_text(s), ngrams)


class TextSim(object):

    def __init__(self, tokenizer=lambda x: x, ngrams=2, smoothing_factor=1):
        self.ngrams = ngrams
        self.tokenizer = tokenizer
        self.smoothing_factor = smoothing_factor

    def fit_model(self, corpus_text):
        tokens = make_ngrams(self.tokenizer(clean_text(corpus_text)), self.ngrams)
        self.token_counts = Counter(tokens)
        self.num_tokens = len(tokens)

    @memoize_method
    def process_clean_string(self, s, ngrams):
        grams = make_ngrams(s, ngrams)
        grams.sort()
        w_grams = [self.num_tokens / (self.token_counts[g] + self.smoothing_factor) for g in grams]
        return grams, w_grams, sum(w_grams)

    def dice_coef(self, a, b):
        if not self.token_counts:
            raise Exception("Fit model before calling similarity")
        if not len(a) or not len(b): return 0.0
        a = clean_text(a)
        b = clean_text(b)
        # quick case for true duplicates
        if a == b: return 1.0
        # if a != b, and a or b are single chars, then they can't possibly match
        if len(a) == 1 or len(b) == 1: return 0.0

        a_bigram_list, w_a, sumwa = self.process_clean_string(a, self.ngrams)
        b_bigram_list, w_b, sumwb = self.process_clean_string(b, self.ngrams)

        lena = len(a_bigram_list)
        lenb = len(b_bigram_list)

        # initialize match counters
        matches = i = j = 0
        while (i < lena and j < lenb):
            if a_bigram_list[i] == b_bigram_list[j]:
                matches += 2 * w_a[i]
                i += 1
                j += 1
            elif a_bigram_list[i] < b_bigram_list[j]:
                i += 1
            else:
                j += 1

        score = float(matches)/float(sumwa + sumwb)
        return score


class IndexedTextSim(object):

    def __init__(self, tokenizer=lambda x: x, ngrams=2, smoothing_factor=1):
        self.ngrams = ngrams
        self.tokenizer = tokenizer
        self.smoothing_factor = smoothing_factor

    def fit_model(self, corpus):
        """
        corpus: list of tuples (id, text)
        """
        self.lookup = [c[0] for c in corpus]
        docs = [tokenize(c[1], self.ngrams) for c in corpus]
        self.dict = corpora.Dictionary(docs)
        bow_corpus = [self.dict.doc2bow(text) for text in docs]
        self.tfidf = models.TfidfModel(bow_corpus)
        self.corpus = [self.tfidf[c] for c in bow_corpus]
        self.index = similarities.docsim.SparseMatrixSimilarity(self.corpus, num_features=len(self.dict))
        self.index.num_best = 3

    @memoize_method
    def all_sim(self, s):
        return self.index[self.to_vector(s)]

    @memoize_method
    def to_vector(self, s):
        return self.tfidf[self.dict.doc2bow(tokenize(s, self.ngrams))]

    @memoize_method
    def most_similar(self, s):
        best_sim = self.all_sim(s)[0]
        return self.lookup[best_sim[0]], best_sim[1]

    def most_similar_batch(self, texts):
        vects = [self.to_vector(t) for t in texts]
        best_sims = self.index[vects]
        results = []
        for sims in best_sims:
            if not sims:
                results.append((None, None))
                continue
            results.append((self.lookup[sims[0][0]], sims[0][1]))
        return results

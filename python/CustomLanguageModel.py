import collections
import math


class CustomLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.trigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.  Compute any counts
        or other corpus statistics in this function."""
        # Unigram counts
        for sentence in corpus.corpus:
            for datum in sentence.data:
                token = datum.word
                self.unigramCounts[token] += 1
                self.total += 1
        self.vocab_size = len(self.unigramCounts)
        # Bigram counts
        for sentence in corpus.corpus:
            if len(sentence) <= 1:
                continue
            previous = sentence.data[0].word
            for datum in sentence.data[1:]:
                token = datum.word
                self.bigramCounts[(previous, token)] += 1
                previous = token
        # Trigram counts
        for sentence in corpus.corpus:
            if len(sentence) <= 2:
                continue
            fst = sentence.data[0].word
            snd = sentence.data[1].word
            for datum in sentence.data[2:]:
                token = datum.word
                self.trigramCounts[(fst, snd, token)] += 1
                fst, snd = snd, token

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability
        of the sentence using your language model. Use whatever data you
        computed in train() here."""
        score = 0.0
        fst = sentence[0]
        snd = sentence[1]
        for token in sentence[2:]:
            tricount = self.trigramCounts[(fst, snd, token)]
            tri_bicount = self.bigramCounts[(fst, snd)]
            bicount = self.bigramCounts[(snd, token)]
            bi_unicount = self.unigramCounts[snd]
            unicount = self.unigramCounts[token]
            if tricount > 0:
                score += math.log(tricount)
                score -= math.log(tri_bicount)
            elif bicount > 0:
                score += math.log(bicount)
                score -= math.log(bi_unicount)
            else:
                score += math.log((unicount + 1))
                score -= math.log(self.total + self.vocab_size)
            fst, snd = snd, token
        return score

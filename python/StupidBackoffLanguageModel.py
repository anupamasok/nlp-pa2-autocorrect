import collections
import math


class StupidBackoffLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
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


    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability
        of the sentence using your language model. Use whatever data you
        computed in train() here."""
        score = 0.0
        previous = sentence[0]
        for token in sentence[1:]:
            bicount = self.bigramCounts[(previous, token)]
            bi_unicount = self.unigramCounts[previous]
            unicount = self.unigramCounts[token]
            if bicount > 0:
                score += math.log(bicount)
                score -= math.log(bi_unicount)
            else:
                score += math.log(0.4)
                score += math.log(unicount + 1)
                score -= math.log(self.total + self.vocab_size)
            previous = token
        return score

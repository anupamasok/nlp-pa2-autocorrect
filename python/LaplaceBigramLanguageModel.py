import collections
import math


class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.  Compute any counts
        or other corpus statistics in this function."""
        for sentence in corpus.corpus:
            previous = sentence.data[0].word
            self.unigramCounts[previous] = self.unigramCounts[previous] + 1
            for datum in sentence.data[1:]:
                token = datum.word
                self.unigramCounts[token] = self.unigramCounts[token] + 1
                self.bigramCounts[(previous, token)] = self.bigramCounts[
                    (previous, token)] + 1
                self.total += 1
                previous = token
        self.vocab_size = len(self.unigramCounts)

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability
        of the sentence using your language model. Use whatever data you
        computed in train() here."""
        score = 0.0
        previous = sentence[0]
        for token in sentence[1:]:
            bicount = self.bigramCounts[(previous, token)]
            unicount = self.unigramCounts[previous]
            score += math.log(bicount + 1)
            score -= math.log(unicount + self.vocab_size)
            previous = token
        return score

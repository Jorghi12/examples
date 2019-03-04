import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.wordfreq = {}

    def add_word(self, word):
        # Store the word frequencies
        if word not in self.wordfreq:
            self.wordfreq[word] = 1
        else:
            self.wordfreq[word] +=1

    def setup_ids(self):
        self.idx2word = []
        self.word2idx = {}

        # Store words by descending frequency
        self.idx2word = sorted(self.wordfreq, key=self.wordfreq.get, reverse=True)

        # Store the word to idx
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # Update the word ids.
        self.dictionary.setup_ids()

        return ids

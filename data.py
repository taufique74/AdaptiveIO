import os
from io import open
import torch
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.total_tokens = 0
        self.counter = Counter()

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def build_vocab(self, path, add_eos=False):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if add_eos:
                    words = line.split() + ['<eos>']
                else:
                    words = line.split()

                # for word in words:
                #     self.add_word(word)
                self.counter.update(words)
                self.total_tokens += len(words)
    
    def sort_vocab(self):
        self.word2idx = {}
        self.idx2word = []

        for word, _ in self.counter.most_common():
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1


    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        train_path = os.path.join(path, 'train.txt')
        test_path = os.path.join(path, 'test.txt')
        valid_path = os.path.join(path, 'valid.txt')

        self.dictionary.build_vocab(train_path)
        self.dictionary.build_vocab(test_path)
        self.dictionary.build_vocab(valid_path)
        
        self.dictionary.sort_vocab()
        
        self.train = self.tokenize(train_path)
        self.valid = self.tokenize(valid_path)
        self.test = self.tokenize(valid_path)
        
    def tokenize(self, path):
        assert os.path.exists(path)
        # self.dictionary.build_vocab(path, add_eos=False)
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split()
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        
        return ids

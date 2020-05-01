import os
from io import open
import torch
from collections import Counter

class Dictionary(object):
    def __init__(self, min_freq=2, add_eos=False):
        self.total_tokens = 0
        self.counter = Counter()
        self.min_freq = min_freq
        self.add_eos = add_eos

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def build_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if self.add_eos:
                    words = line.split() + ['<eos>']
                else:
                    words = line.split()

                self.counter.update(words)
                self.total_tokens += len(words)
    
    def sort_vocab(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        if self.add_eos:
            self.word2idx = {'<eos>': 1}
            self.idx2word.append('<eos>')

        for word, frequency in self.counter.most_common():
            if frequency >= self.min_freq:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
    
    def count_from_freq(self, files, min_freq):
          dictionary = Counter()
          for filename in files:
              with open(filename, 'r', encoding='utf8') as f:
                  for line in f:
                      dictionary.update(line.split())

          count = 0
          for _, freq in dictionary.most_common():
              if freq >= min_freq:
                  count += 1
          
          return count

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

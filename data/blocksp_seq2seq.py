from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import numpy as np
from numba import jit
import pickle


class DatasetBlocksp(Dataset):
    def __init__(self, corpus_path, seq_len):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        # self.vocab['mask_index'] = 21
        self.vocab['sos_index'] = 21
        # self.vocab['eos_index'] = 23
        self.vocab['unk_index'] = 22

        self.seq_len = seq_len
        self.corpus_path = corpus_path

        # self.chunksize = 10**4
        # self.df = pd.read_csv(corpus_path, sep=',', chunksize=self.chunksize, engine='python')
        #                       # dtype={'seq_num': np.int32, 'block_a': str, 'block_b': str, 'fam': str})
        # self.buffer = self.df.__next__()
        # self.count = 0
        # self.seq_a, self.seq_b = self.buffer['block_a'].values, self.buffer['block_b'].values
        #
        # seq_num = pd.read_csv(corpus_path, usecols=['seq_num'])
        # self.num_seq = seq_num.shape[0]

        df = pd.read_csv(corpus_path, usecols=['block_a', 'block_b'])
        self.num_seq = df.shape[0]
        self.seq_a, self.seq_b = df['block_a'].values, df['block_b'].values

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        # print(item)
        # item = item - self.count * self.chunksize
        # if item >= self.chunksize:
        #     self.buffer = self.df.__next__()
        #     self.seq_a, self.seq_b = self.buffer['block_a'].values, self.buffer['block_b'].values
        #     self.count += 1
        #     item -= self.chunksize
        # if item >= self.chunksize:
        #     raise ValueError('item is outside the next chunk')

        t1, t2 = self.seq_a[item], self.seq_b[item]

        t1 = t1.upper()
        t2 = t2.upper()

        src = self.tokenizer(t1)
        tgt = self.tokenizer(t2)
        tgt_x = [self.vocab['sos_index']] + tgt[:-1]

        output = {"src": src,
                  "tgt_x": tgt_x,
                  "tgt_y": tgt}

        for key, value in output.items():
            output[key] = torch.tensor(value, requires_grad=False)
        return output

    def tokenizer(self, sentence):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            try:
                tokens[i] = self.vocab[token]
            except KeyError:
                tokens[i] = self.vocab['unk_index']
        # crop long seq or pad short seq
        if len(tokens) > self.seq_len:
            tokens = tokens[0:self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(tokens))]
            tokens.extend(padding)
        return tokens


if __name__ == '__main__':
    dataset = DatasetBlocksp('data/seq_split_small.csv', seq_len=50)




from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import numpy as np
from numba import jit
import pickle


class DatasetBlock(Dataset):
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

        with open(corpus_path, "r") as f:
            self.seq = [line[:-1] for line in f]
        self.num_seq = len(self.seq)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        t1, t2 = self.seq[item].split('$')

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
    dataset = DatasetBlock('data/seq_block_pair_small.txt', seq_len=32)




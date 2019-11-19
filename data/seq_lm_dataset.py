from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import numpy as np
from numba import jit
import pickle


class DatasetSeqLM(Dataset):
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
        # self.corpus_path = corpus_path

        df = pd.read_csv(corpus_path)
        # df.sort_values(by='seq', ascending=True, inplace=True)
        self.seq = df['seq_unalign'].values
        self.slen = df['seq_len'].values
        self.num_seq = len(self.slen)

        # with open(corpus_path, "r") as f:
        #     self.seq = [line[:-1] for line in f]
        # self.num_seq = len(self.seq)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        t1 = self.seq[item]

        seq = self.tokenizer(t1)
        seq_x = [self.vocab['sos_index']] + seq[:-1]

        output = {"seq": seq,
                  "seq_x": seq_x,
                  }

        for key, value in output.items():
            # output[key] = torch.tensor(value, requires_grad=False)
            output[key] = torch.tensor(value)
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
    # dataset = DatasetSeqLM('data/pf00400_unalign_clean_train.txt', seq_len=50)
    dataset = DatasetSeqLM('data/seq_unalign_indel_all_cut_sample.csv', seq_len=256)




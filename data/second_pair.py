from torch.utils.data import Dataset
import tqdm
import torch
import random
import pandas as pd
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, corpus_path, seq_len=12, encoding="utf-8", corpus_lines=10, on_memory=True):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        self.vocab['start_index'] = 21
        # self.vocab['mask_index'] = 21
        # self.vocab['unk_index'] = 22
        # self.vocab['eos_index'] = 23
        # self.vocab['sos_index'] = 24

        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            # if self.corpus_lines is None and not on_memory:
            #     for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
            #         self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            # self.random_file = open(corpus_path, "r", encoding=encoding)
            # for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
            #     self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2 = self.get_corpus_line(item)
        src = self.tokenizer(t1)
        tgt = self.tokenizer(t2)
        tgt_x = [self.vocab['start_index']] + tgt[:-1]

        output = {"src": src,
                  "tgt_x": tgt_x,
                  "tgt_y": tgt}
        # print(output)
        return {key: torch.tensor(value) for key, value in output.items()}

    def tokenizer(self, sentence):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab[token]
        # crop long seq or pad short seq
        if len(tokens) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            start_idx = 0
            tokens = tokens[start_idx:start_idx+self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(tokens))]
            tokens.extend(padding)
        return tokens

    def get_corpus_line(self, item):
        if self.on_memory:
            line = self.lines[item]
            t1, t2 = line.split(sep='$')
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
            t1, t2 = line[:-1].split(sep='$')
        return t1, t2


class TranslationDataset(Dataset):
    def __init__(self, corpus_path, seq_len=12, encoding="utf-8", corpus_lines=10):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        self.vocab['start_index'] = 21
        # self.vocab['mask_index'] = 21
        # self.vocab['unk_index'] = 22
        # self.vocab['eos_index'] = 23
        # self.vocab['sos_index'] = 24

        self.idx2vocab = {y: x for x, y in self.vocab.items()}
        self.seq_len = seq_len

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = [line[:-1]
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        src = self.tokenizer(self.lines[item])
        return torch.tensor(src)

    def tokenizer(self, sentence):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab[token]
        # crop long seq or pad short seq
        if len(tokens) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            start_idx = 0
            tokens = tokens[start_idx:start_idx+self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(tokens))]
            tokens.extend(padding)
        return tokens

    def idx2word(self, idx):
        return self.idx2vocab[idx]


if __name__ == '__main__':
    # train_dataset = BERTDataset('data/pair_4A_proc.txt')
    trans_dataset = TranslationDataset('data/pair_6A_translate_src.txt')




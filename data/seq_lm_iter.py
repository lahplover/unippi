from torch.utils.data import IterableDataset, DataLoader
import torch
import random
import pandas as pd
import numpy as np
from numba import jit
import pickle


class DatasetSeqLMIter(IterableDataset):
    def __init__(self, corpus_path):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        # self.vocab['mask_index'] = 21
        self.vocab['sos_index'] = 21
        # self.vocab['eos_index'] = 23
        self.vocab['unk_index'] = 22

        # self.seq_len = seq_len
        # self.corpus_path = corpus_path

        df = pd.read_csv(corpus_path)
        # df.sort_values(by='seq', ascending=True, inplace=True)
        self.seq = df['seq_unalign'].values
        self.slen = df['seq_len'].values
        self.num_seq = len(self.slen)

        self.seq_len_list = [256, 512, 1000]
        self.batch_size_list = [200, 100, 50]
        self.start_list = [0, 1015636, 1424581, self.num_seq]

    def __len__(self):
        return self.num_seq

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._get_batch_iter()
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # print(num_workers, worker_id)
            return self._get_batch_iter(num_workers, worker_id)

    def _get_batch_iter(self, num_workers=0, worker_id=0):
        for k in range(len(self.batch_size_list)):
            seq_len = self.seq_len_list[k]
            batch_size = self.batch_size_list[k]
            start = self.start_list[k]
            end = self.start_list[k+1]
            if num_workers <= 1:
                for i in range(start, end-batch_size, batch_size):
                    yield self._get_batch(i, batch_size, seq_len)
            else:
                for i in range(start, end-batch_size*num_workers, batch_size*num_workers):
                    yield self._get_batch(i+batch_size*worker_id, batch_size, seq_len)

    def _get_batch(self, i, batch_size, seq_len):
        seq = []
        end = min(i+batch_size, self.num_seq)
        for item in range(i, end):
            t1 = self.seq[item]
            t1 = np.array(self.tokenizer(t1, seq_len))
            # print(t1)
            seq.append(t1)

        seq = np.vstack(seq)
        # print(seq)
        seq_x = np.pad(seq[:, :-1], ((0, 0), (1, 0)), 'constant', constant_values=self.vocab['sos_index'])

        output = {"seq": seq,
                  "seq_x": seq_x,
                  }

        for key, value in output.items():
            # output[key] = torch.tensor(value, requires_grad=False)
            output[key] = torch.tensor(value)
        return output

    def tokenizer(self, sentence, seq_len):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            try:
                tokens[i] = self.vocab[token]
            except KeyError:
                tokens[i] = self.vocab['unk_index']
        # crop long seq or pad short seq
        if len(tokens) > seq_len:
            tokens = tokens[0:seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(seq_len - len(tokens))]
            tokens.extend(padding)
        return tokens


if __name__ == '__main__':
    dataset = DatasetSeqLMIter('data/multinode/uniref50_train_shuffle_0_iter.csv')

    data_loader = DataLoader(dataset, num_workers=2)

    for i, data in enumerate(data_loader):
        if i % 1000 == 0:
            print(i)




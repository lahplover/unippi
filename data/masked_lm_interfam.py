from torch.utils.data import Dataset
import tqdm
import torch
import random
import pandas as pd
import numpy as np


class DatasetInterDomain(Dataset):
    def __init__(self, corpus_path, seq_len, seq_mode='one',
                 relative_3d=False, relative_3d_size=10, relative_3d_step=2
                 ):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        self.vocab['mask_index'] = 21
        self.vocab['sos_index'] = 22
        self.vocab['eos_index'] = 23
        self.vocab['unk_index'] = 24

        self.seq_len = seq_len
        self.seq_mode = seq_mode
        self.corpus_path = corpus_path
        self.relative_3d = relative_3d
        self.relative_3d_size = relative_3d_size
        self.relative_3d_step = relative_3d_step
        self.max_relative_3d = relative_3d_size * relative_3d_step
        self.vocab_3d = {x: x for x in np.arange(relative_3d_size+1)}
        self.vocab_3d['no_msa'] = relative_3d_size + 1
        self.vocab_3d['pad_index'] = relative_3d_size + 2
        self.vocab_3d['sos_index'] = relative_3d_size + 3
        self.vocab_3d['eos_index'] = relative_3d_size + 4
        self.vocab_3d['inter_12'] = relative_3d_size + 5

        # with open(corpus_path, "r", encoding=encoding) as f:
        #     # if self.corpus_lines is None and not on_memory:
        #     #     for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
        #     #         self.corpus_lines += 1
        #
        #     self.lines = [line[:-1]
        #                   for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
        #     self.corpus_lines = len(self.lines)

        df = pd.read_pickle(corpus_path)
        # df.sort_values(by='seq', ascending=True, inplace=True)
        self.seq_a = df['seg_a'].values
        self.seq_b = df['seg_b'].values

        # self.slen = df['seq_len'].values
        self.num_seq = len(self.seq_a)
        # self.num_item_two = int(0.1 * self.num_seq)

        if relative_3d:
            self.dist_mat_array = df['dist_mat'].values
            self.seg_order_in_dm = df['seg_order_in_dm'].values

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        output = self._get_item_two(item)
        for key, value in output.items():
            output[key] = torch.tensor(value, requires_grad=False)
        return output

    def _get_item_two(self, item):
        if random.random() > 0.5:
            is_next_label = 1
            t1, t2 = self.seq_a[item], self.seq_b[item]
            if self.relative_3d:
                dist_mat = self._get_clipped_dist_mat(item)
                if self.seg_order_in_dm[item] == 'ab':
                    t1_dist_mat = dist_mat[:len(t1), :len(t1)]
                    t2_dist_mat = dist_mat[len(t1):, len(t1):]
                else:
                    t2_dist_mat = dist_mat[:len(t2), :len(t2)]
                    t1_dist_mat = dist_mat[len(t2):, len(t2):]
        else:
            is_next_label = 0
            # fixed bug, add _t1 and _t2
            item2 = np.random.randint(0, self.num_seq)
            t1, _t2 = self.seq_a[item], self.seq_b[item]
            _t1, t2 = self.seq_a[item2], self.seq_b[item2]
            if self.relative_3d:
                dist_mat = self._get_clipped_dist_mat(item)
                if self.seg_order_in_dm[item] == 'ab':
                    t1_dist_mat = dist_mat[:len(t1), :len(t1)]
                else:
                    t1_dist_mat = dist_mat[len(_t2):, len(_t2):]
                dist_mat = self._get_clipped_dist_mat(item2)
                if self.seg_order_in_dm[item2] == 'ab':
                    t2_dist_mat = dist_mat[len(_t1):, len(_t1):]
                else:
                    t2_dist_mat = dist_mat[:len(t2), :len(t2)]

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab['sos_index']] + t1_random + [self.vocab['eos_index']]
        t2 = t2_random + [self.vocab['eos_index']]

        t1_label = [self.vocab['pad_index']] + t1_label + [self.vocab['pad_index']]
        t2_label = t2_label + [self.vocab['pad_index']]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])
        bert_input = (t1 + t2)
        bert_label = (t1_label + t2_label)
        if len(bert_input) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            start_idx = 0
            bert_input = bert_input[start_idx:start_idx+self.seq_len]
            bert_label = bert_label[start_idx:start_idx+self.seq_len]
            segment_label = segment_label[start_idx:start_idx+self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        if self.relative_3d:
            t1_dist_mat = np.pad(t1_dist_mat, ((1, 1), (1, 1)), 'constant',
                                 constant_values=((self.vocab_3d['sos_index'], self.vocab_3d['eos_index']),
                                                  (self.vocab_3d['sos_index'], self.vocab_3d['eos_index'])))
            t2_dist_mat = np.pad(t2_dist_mat, ((0, 1), (0, 1)), 'constant', constant_values=self.vocab_3d['eos_index'])

            t12_len = t1_dist_mat.shape[0] + t2_dist_mat.shape[0]
            t12_dist_mat = np.ones((t12_len, t12_len), dtype=int) * self.vocab_3d['inter_12']
            t12_dist_mat[:t1_dist_mat.shape[0], :t1_dist_mat.shape[0]] = t1_dist_mat
            t12_dist_mat[t1_dist_mat.shape[0]:, t1_dist_mat.shape[0]:] = t2_dist_mat

            # crop long seq or pad for short seq
            if t12_len >= self.seq_len:
                # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
                start_idx = 0
                t12_dist_mat = t12_dist_mat[start_idx:start_idx + self.seq_len, start_idx:start_idx + self.seq_len]
            else:
                pad_dist_mat = self.seq_len - t12_len
                t12_dist_mat = np.pad(t12_dist_mat, ((0, pad_dist_mat), (0, pad_dist_mat)),
                                      'constant', constant_values=self.vocab_3d['pad_index'])
            assert (t12_dist_mat.shape[0] == self.seq_len)
        else:
            t12_dist_mat = 0

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "dist_mat": t12_dist_mat}
        return output

    def _get_clipped_dist_mat(self, item):
        t1_dist_mat = self.dist_mat_array[item]
        # t1_dist_mat[t1_dist_mat == -1] = -1 * self.relative_3d_step  # positions with no distances from MSA
        # t1_dist_mat_clipped = torch.clamp(t1_dist_mat, max=self.max_relative_3d)
        t1_dist_mat_clipped = np.clip(t1_dist_mat, a_min=None, a_max=self.max_relative_3d)
        t1_dist_mat = (t1_dist_mat_clipped // self.relative_3d_step).astype(int)
        # t1_dist_mat[t1_dist_mat == -1] = self.vocab_3d['no_msa']  # positions with no distances from MSA
        return t1_dist_mat

    def random_word(self, sentence):
        # print(sentence)
        tokens = list(sentence)
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab['mask_index']

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab[token]

                output_label.append(self.vocab[token])

            else:
                tokens[i] = self.vocab[token]
                output_label.append(0)

        return tokens, output_label

    # def get_corpus_line(self, item):
    #     if self.on_memory:
    #         return self.lines[item]
    #     else:
    #         line = self.file.__next__()
    #         if line is None:
    #             self.file.close()
    #             self.file = open(self.corpus_path, "r", encoding=self.encoding)
    #             line = self.file.__next__()
    #
    #         t1 = line[:-1]
    #         return t1


if __name__ == '__main__':
    train_dataset = DatasetInterDomain('data/pdb_interDM_unique_diff_domain_sample.pkl',
                                       seq_len=256, seq_mode='two',
                                       relative_3d=True)




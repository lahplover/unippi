from torch.utils.data import Dataset
import tqdm
import torch
import random
import pandas as pd
import numpy as np


class DatasetPDBInterDomain(Dataset):
    def __init__(self, corpus_path, seq_len, encoding="utf-8",
                 relative_3d=False, relative_3d_size=10, relative_3d_step=2,
                 corpus_lines=10, on_memory=True):
        # self.vocab = vocab
        amino_acids = pd.read_csv('data/amino_acids.csv')
        self.vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
        self.vocab['pad_index'] = 0
        self.vocab['mask_index'] = 21
        self.vocab['sos_index'] = 22
        self.vocab['eos_index'] = 23
        self.vocab['unk_index'] = 24

        self.seq_len = seq_len
        self.on_memory = on_memory
        self.num_seq = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
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
        self.seg_a = df['seg_a'].values
        self.seg_b = df['seg_b'].values

        self.num_seq = df.shape[0]

        if relative_3d:
            # df_dist_mat = pd.read_pickle('data/pf00400_dist_mat.pkl.gz',  compression='gzip')
            # self.dist_mat_dict = {x: y for x, y in zip(df_dist_mat['seq'], df_dist_mat['dist_mat'])}
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
        t1, t2 = self.seg_a[item], self.seg_b[item]
        dist_mat = self._get_clipped_dist_mat(item)
        # order of t1, t2 in dist_mat? t1-t2  or t2-t1?
        if self.seg_order_in_dm[item] == 'ab':
            t1_dist_mat = dist_mat[:len(t1), :len(t1)]
            t2_dist_mat = dist_mat[len(t1):, len(t1):]
            dist_mat_inter = dist_mat[:len(t1), len(t1):]   # shape (t1, t2)
        else:
            t2_dist_mat = dist_mat[:len(t2), :len(t2)]
            t1_dist_mat = dist_mat[len(t2):, len(t2):]
            dist_mat_inter = dist_mat[len(t2):, :len(t2)]   # shape (t1, t2)
        # print(t1_dist_mat.shape, t2_dist_mat.shape, len(t1), len(t2))
        t1 = self.tokenizer(t1)
        t2 = self.tokenizer(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab['sos_index']] + t1 + [self.vocab['eos_index']]
        t2 = t2 + [self.vocab['eos_index']]

        t1_dist_mat = np.pad(t1_dist_mat, ((1, 1), (1, 1)), 'constant',
                             constant_values=((self.vocab_3d['sos_index'], self.vocab_3d['eos_index']),
                                              (self.vocab_3d['sos_index'], self.vocab_3d['eos_index'])))
        t2_dist_mat = np.pad(t2_dist_mat, ((0, 1), (0, 1)), 'constant', constant_values=self.vocab_3d['eos_index'])

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])

        seq_input = (t1 + t2)
        t12_len = len(seq_input)
        dist_mat_input = np.ones((t12_len, t12_len), dtype=int) * self.vocab_3d['inter_12']
        dist_mat_input[:t1_dist_mat.shape[0], :t1_dist_mat.shape[0]] = t1_dist_mat
        dist_mat_input[t1_dist_mat.shape[0]:, t1_dist_mat.shape[0]:] = t2_dist_mat

        dist_mat_target = np.ones((t12_len, t12_len), dtype=int) * self.vocab_3d['no_msa']
        dist_mat_target[1:t1_dist_mat.shape[0]-1, t1_dist_mat.shape[0]:-1] = dist_mat_inter

        if len(seq_input) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            seq_input = seq_input[0:self.seq_len]
            segment_label = segment_label[0:self.seq_len]
            dist_mat_input = dist_mat_input[0:self.seq_len, 0:self.seq_len]
            dist_mat_target = dist_mat_target[0:self.seq_len, 0:self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(seq_input))]
            seq_input.extend(padding), segment_label.extend(padding)

            pad_dist_mat = self.seq_len - t12_len
            dist_mat_input = np.pad(dist_mat_input, ((0, pad_dist_mat), (0, pad_dist_mat)),
                                    'constant', constant_values=self.vocab_3d['pad_index'])
            dist_mat_target = np.pad(dist_mat_target, ((0, pad_dist_mat), (0, pad_dist_mat)),
                                     'constant', constant_values=self.vocab_3d['no_msa'])

        output = {"seq_input": seq_input,
                  "dist_mat_input": dist_mat_input,
                  "segment_label": segment_label,
                  "dist_mat_target": dist_mat_target}
        return output

    def _get_clipped_dist_mat(self, item):
        t1_dist_mat = self.dist_mat_array[item]
        # t1_dist_mat[t1_dist_mat == -1] = -1 * self.relative_3d_step  # positions with no distances from MSA
        # t1_dist_mat_clipped = torch.clamp(t1_dist_mat, max=self.max_relative_3d)
        t1_dist_mat_clipped = np.clip(t1_dist_mat, a_min=None, a_max=self.max_relative_3d)
        t1_dist_mat = (t1_dist_mat_clipped // self.relative_3d_step).astype(int)
        # t1_dist_mat[t1_dist_mat == -1] = self.vocab_3d['no_msa']  # positions with no distances from MSA
        return t1_dist_mat

    def tokenizer(self, sentence):
        # print(sentence)
        tokens = list(sentence)

        for i, token in enumerate(tokens):
            try:
                tokens[i] = self.vocab[token]
            except KeyError:
                tokens[i] = self.vocab['unk_index']

        return tokens


if __name__ == '__main__':
    train_dataset = DatasetPDBInterDomain('data/pdb_interDM_unique_diff_domain_sample.pkl', seq_len=256, relative_3d=True)



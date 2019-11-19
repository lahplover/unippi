from torch.utils.data import Dataset
import tqdm
import torch
import random
import pandas as pd
import numpy as np
from numba import jit
import pickle


@jit(nopython=True)
def _get_dist_mat_inner_loop(sindel, h, dist_mat, dist_mat_hmm):
    count_hmm_i = 0
    count_seq_i = 0
    # print(seq)
    for i in range(h):
        si = sindel[i]
        if si == 1:
            count_seq_i += 1
        elif si == 2:
            count_hmm_i += 1
        else:
            count_hmm_j = 0
            count_seq_j = 0
            for j in range(h):
                sj = sindel[j]
                if sj == 1:
                    count_seq_j += 1
                elif sj == 2:
                    count_hmm_j += 1
                else:
                    # print(count_seq_i, count_seq_j, count_hmm_i, count_hmm_j)
                    dist_mat[count_seq_i, count_seq_j] = dist_mat_hmm[count_hmm_i, count_hmm_j]
                    count_hmm_j += 1
                    count_seq_j += 1
            count_hmm_i += 1
            count_seq_i += 1
    return dist_mat


class DatasetProtEng(Dataset):
    def __init__(self, corpus_path, seq_len, seq_mode='one',
                 relative_3d=False, relative_3d_size=10, relative_3d_step=2,
                 regression=True
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

        df = pd.read_csv(corpus_path)
        # df.sort_values(by='seq', ascending=True, inplace=True)
        self.seq = df['primary'].values
        self.slen = df['protein_length'].values
        if regression:
            self.flux = df['log_fluorescence'].values
        else:
            flux_class = np.zeros(df.shape[0], dtype=int)
            flux_class[df['log_fluorescence'].values > 2.5] = 1
            self.flux = flux_class
            # self.flux = np.round(np.clip(df['log_fluorescence'].values, a_min=0, a_max=5)).astype(np.int)
        self.num_seq = len(self.slen)

        if relative_3d:
            self.seq_indel = df['seq_indel'].values
            self.fam = df['pfam_acc'].values
            # df_dist_mat = pd.read_pickle('data/pf00400_dist_mat.pkl.gz',  compression='gzip')
            # self.dist_mat_dict = {x: y for x, y in zip(df_dist_mat['seq'], df_dist_mat['dist_mat'])}
            # self.dist_mat_dict = {x: y for x, y in zip(df['seq'], df['dist_mat'])}

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        output = self._get_item_one(item)
        for key, value in output.items():
            output[key] = torch.tensor(value, requires_grad=False)
        return output

    def _get_item_one(self, item):
        t1 = self.seq[item]

        t1 = self.tokenizer(t1)
        seq_input = [self.vocab['sos_index']] + t1

        # crop long seq or pad short seq
        if len(seq_input) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            start_idx = 0
            seq_input = seq_input[start_idx:start_idx+self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(seq_input))]
            seq_input.extend(padding)
        assert(len(seq_input) == self.seq_len)

        if self.relative_3d:
            t1_fam = self.fam[item]
            t1_indel = self.seq_indel[item]
            t1_dist_mat = self._get_clipped_dist_mat(t1_fam, t1, t1_indel)
            # pad for 'sos_index'
            t1_dist_mat = np.pad(t1_dist_mat, ((1, 0), (1, 0)), 'constant', constant_values=self.vocab_3d['sos_index'])
            # crop long seq or pad for short seq
            if t1_dist_mat.shape[0] >= self.seq_len:
                # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
                start_idx = 0
                t1_dist_mat = t1_dist_mat[start_idx:start_idx+self.seq_len, start_idx:start_idx+self.seq_len]
            else:
                pad_dist_mat = self.seq_len - t1_dist_mat.shape[0]
                t1_dist_mat = np.pad(t1_dist_mat, ((0, pad_dist_mat), (0, pad_dist_mat)),
                                     'constant', constant_values=self.vocab_3d['pad_index'])
            assert (t1_dist_mat.shape[0] == self.seq_len)
        else:
            t1_dist_mat = 0
        output = {"seq_input": seq_input,
                  "flux": self.flux[item],
                  "dist_mat": t1_dist_mat}
        # print(output)
        return output

    def tokenizer(self, sentence):
        # print(sentence)
        tokens = list(sentence)

        for i, token in enumerate(tokens):
            try:
                tokens[i] = self.vocab[token]
            except KeyError:
                tokens[i] = self.vocab['unk_index']

        return tokens

    def _get_clipped_dist_mat(self, fam, seq, seq_indel):
        with open(f'/share/home/yangh/bio/pfam/pfam_pdb_distmat/{fam}/hmm_distmat_{fam}.pkl', 'rb') as distmat_pkl:
            dist_mat_hmm = pickle.load(distmat_pkl)

        dist_mat = np.ones((len(seq), len(seq))) * (-1)
        sindel = np.zeros(len(seq_indel), dtype=int)
        for i, si in enumerate(seq_indel):
            if si.islower():
                sindel[i] = 1
            elif si == '-':
                sindel[i] = 2
        dist_mat = _get_dist_mat_inner_loop(sindel, sindel.shape[0], dist_mat, dist_mat_hmm)

        # convert distmat into int grid values [-1, 0, 1, ..., 19, 20] * (2A)
        dist_mat[dist_mat == 0] = -1 * self.relative_3d_step  # positions with no distances from MSA
        dist_mat[dist_mat == -1] = -1 * self.relative_3d_step  # positions with no distances from MSA
        dist_mat_clipped = np.clip(dist_mat, a_min=None, a_max=self.max_relative_3d)
        # t1_dist_mat_clipped = torch.clamp(t1_dist_mat, max=self.max_relative_3d)
        # dist_mat = np.round(dist_mat / self.relative_3d_step)
        dist_mat_key = (dist_mat_clipped // self.relative_3d_step).astype(int)
        dist_mat_key[dist_mat_key == -1] = self.vocab_3d['no_msa']  # positions with no distances from MSA
        return dist_mat_key


if __name__ == '__main__':
    dataset = DatasetProtEng('data/fluorescence_train.csv', seq_len=256,
                               seq_mode='one', relative_3d=False)




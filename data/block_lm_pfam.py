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


@jit(nopython=True)
def _dm_peak_one(contact_mat):
    n = contact_mat.shape[0]
    ij_list = []
    for i in range(-16, n-48, 16):
        for j in range(i+32, n-16, 16):
            if np.sum(contact_mat[i:i+32, j:j+32]) >= 10:
                ij_list.append((i, j))
    return ij_list


class DatasetSeqBlock(Dataset):
    def __init__(self, corpus_path, seq_len,
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
        self.seq = df['seq_unalign'].values
        self.fam = df['pfam_acc'].values
        self.num_seq = len(self.fam)
        # self.dmpeak = df['dmpeak'].values
        # df_dmpeak = pd.read_pickle(dmpeak_data_path)
        # self.dmpeak = df_dmpeak['dmpeak'].values

        self.seq_indel = df['seq_indel'].values
        df_dm_hmm = pd.read_pickle('data/distmat_hmm_pfam_all.pkl')
        # all pfam_acc in self.fam have one item in distmat_hmm_pfam_all.pkl
        self.dist_mat_dict = {x: y for x, y in zip(df_dm_hmm['fam'], df_dm_hmm['distmat'])}

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        output = self._get_item_two(item)
        for key, value in output.items():
            output[key] = torch.tensor(value, requires_grad=False)
        return output

    def _get_item_two(self, item):
        t12_fam = self.fam[item]
        t12 = self.seq[item]
        t12_indel = self.seq_indel[item]
        dist_mat = self._get_clipped_dist_mat(t12_fam, t12, t12_indel)

        n = dist_mat.shape[0]
        contact_mat = np.zeros((n, n), dtype=np.int8)
        contact_mat[(dist_mat > 0) & (dist_mat < 8 // self.relative_3d_step)] = 1
        dmpeak = _dm_peak_one(contact_mat)

        if len(dmpeak) > 0:
            i, j = dmpeak[np.random.randint(0, len(dmpeak))]
        else:
            i, j = (0, 32)

        t1, t2 = t12[i:i+32], t12[j:j+32]

        t1 = self.tokenizer(t1)
        t2 = self.tokenizer(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab['sos_index']] + t1 + [self.vocab['eos_index']]
        t2 = t2 + [self.vocab['eos_index']]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])

        seq_input = (t1 + t2)
        t12_len = len(seq_input)

        if self.relative_3d:
            t1_dist_mat = dist_mat[i:i+32, i:i+32]
            t2_dist_mat = dist_mat[j:j+32, j:j+32]
            dist_mat_inter = dist_mat[i:i+32, j:j+32]
            # print(dist_mat_inter)

            t1_dist_mat = np.pad(t1_dist_mat, ((1, 1), (1, 1)), 'constant',
                                 constant_values=((self.vocab_3d['sos_index'], self.vocab_3d['eos_index']),
                                                  (self.vocab_3d['sos_index'], self.vocab_3d['eos_index'])))
            t2_dist_mat = np.pad(t2_dist_mat, ((0, 1), (0, 1)), 'constant', constant_values=self.vocab_3d['eos_index'])

            dist_mat_input = np.ones((t12_len, t12_len), dtype=int) * self.vocab_3d['no_msa']
            dist_mat_input[1:t1_dist_mat.shape[0]-1, 1:t1_dist_mat.shape[0]-1] = t1_dist_mat[1:-1, 1:-1]
            dist_mat_input[t1_dist_mat.shape[0]:-1, t1_dist_mat.shape[0]:-1] = t2_dist_mat[:-1, :-1]
            dist_mat_input[1:t1_dist_mat.shape[0]-1, t1_dist_mat.shape[0]:-1] = dist_mat_inter
            dist_mat_input[t1_dist_mat.shape[0]:-1, 1:t1_dist_mat.shape[0]-1] = dist_mat_inter.T
        else:
            dist_mat_input = 0

        if len(seq_input) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            seq_input = seq_input[0:self.seq_len]
            segment_label = segment_label[0:self.seq_len]
            if self.relative_3d:
                dist_mat_input = dist_mat_input[0:self.seq_len, 0:self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(seq_input))]
            seq_input.extend(padding), segment_label.extend(padding)
            pad_dist_mat = self.seq_len - t12_len
            if self.relative_3d:
                dist_mat_input = np.pad(dist_mat_input, ((0, pad_dist_mat), (0, pad_dist_mat)),
                                        'constant', constant_values=self.vocab_3d['pad_index'])

        seq_target = seq_input[1:] + [self.vocab['eos_index']]

        output = {"seq_input": seq_input,
                  "seq_target": seq_target,
                  "segment_label": segment_label,
                  "dist_mat_input": dist_mat_input,
                  }
        return output

    def _get_clipped_dist_mat(self, fam, seq, seq_indel):
        # with open(f'/share/home/yangh/bio/pfam/pfam_pdb_distmat/{fam}/hmm_distmat_{fam}.pkl', 'rb') as distmat_pkl:
        #     dist_mat_hmm = pickle.load(distmat_pkl)
        dist_mat_hmm = self.dist_mat_dict[fam]

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

    def tokenizer(self, sentence):
        # print(sentence)
        tokens = list(sentence)

        for i, token in enumerate(tokens):
            try:
                tokens[i] = self.vocab[token]
            except KeyError:
                tokens[i] = self.vocab['unk_index']

        return tokens


def plot_dm(dataset):
    import matplotlib.pyplot as pl
    # for i in range(len(dataset)):
    for i in range(0, len(dataset), 1000):
        a = dataset[i]
        img = np.vstack((a['dist_mat_input'], a['dist_mat_target']))
        fig = pl.figure()
        pl.imshow(img)
        pl.colorbar()
        pl.savefig(f'data/fig_dm/{i}.pdf')
        pl.close(fig)

    for i in range(0, len(dataset), 1000):
        a = dataset[i]
        img = a['dist_mat_target']
        fig = pl.figure()
        pl.imshow(img)
        pl.colorbar()
        pl.savefig(f'data/fig_dm/1d_{i}.pdf')
        pl.close(fig)


if __name__ == '__main__':
    dataset = DatasetSeqBlock('data/seq_unalign_indel_all_dmpeak_sample.csv', seq_len=67, relative_3d=True)




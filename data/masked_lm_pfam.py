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


class DatasetSeq(Dataset):
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

        df = pd.read_csv(corpus_path)
        # df.sort_values(by='seq', ascending=True, inplace=True)
        self.seq = df['seq_unalign'].values
        self.slen = df['seq_len'].values
        self.num_seq = len(self.slen)
        self.num_item_two = int(0.1 * self.num_seq)

        if relative_3d:
            self.seq_indel = df['seq_indel'].values
            self.fam = df['pfam_acc'].values
            # df_dist_mat = pd.read_pickle('data/pf00400_dist_mat.pkl.gz',  compression='gzip')
            # self.dist_mat_dict = {x: y for x, y in zip(df_dist_mat['seq'], df_dist_mat['dist_mat'])}
            # self.dist_mat_dict = {x: y for x, y in zip(df['seq'], df['dist_mat'])}

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        if self.seq_mode == 'one':
            output = self._get_item_one(item)
        elif self.seq_mode == 'two':
            output = self._get_item_two(item)
        else:
            raise ValueError('seq_mode should be one/two.')
        for key, value in output.items():
            output[key] = torch.tensor(value, requires_grad=False)
        return output

    def _get_item_one(self, item):
        t1 = self.seq[item]

        t1_random, t1_label = self.random_word(t1)
        bert_input = [self.vocab['sos_index']] + t1_random
        bert_label = [self.vocab['pad_index']] + t1_label

        # crop long seq or pad short seq
        if len(bert_input) > self.seq_len:
            # start_idx = np.random.randint(0, len(bert_input) - self.seq_len)
            start_idx = 0
            bert_input = bert_input[start_idx:start_idx+self.seq_len]
            bert_label = bert_label[start_idx:start_idx+self.seq_len]
        else:
            padding = [self.vocab['pad_index'] for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding)
        assert(len(bert_input) == self.seq_len)

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
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "dist_mat": t1_dist_mat}
        # print(output)
        return output

    def _get_item_two(self, item):
        if random.random() > 0.5:
            is_next_label = 1
            t12 = self.seq[item]
            t12_len = len(t12)
            i_split = np.random.randint(int(t12_len/3), int(t12_len*2/3))
            t1, t2 = t12[:i_split], t12[i_split:]
            if self.relative_3d:
                t12_fam = self.fam[item]
                t12_indel = self.seq_indel[item]
                t12_dist_mat = self._get_clipped_dist_mat(t12_fam, t12, t12_indel)
                t1_dist_mat = t12_dist_mat[:i_split, :i_split]
                t2_dist_mat = t12_dist_mat[i_split:, i_split:]
        else:
            is_next_label = 0
            t1a = self.seq[item]
            t1a_len = self.slen[item]
            t1_split = np.random.randint(int(t1a_len/3), int(t1a_len*2/3))

            # get another longer sequence, cut the tail so that len(t1_tail) = len(t2_tail)
            # ind = (self.slen >= t1a_len) & (self.slen < t1a_len + 20)
            # seq2 = self.seq[ind]
            # t2a = seq2[random.randrange(len(seq2))]
            item2 = item + np.random.randint(1, self.num_item_two)
            item2 = min(item2, self.num_seq-1)
            t2a = self.seq[item2]
            t2a_len = len(t2a)
            t2_split = t2a_len - (t1a_len - t1_split)
            t1, t2 = t1a[:t1_split], t2a[t2_split:]
            if self.relative_3d:
                t1a_fam = self.fam[item]
                t1a_indel = self.seq_indel[item]
                t1a_dist_mat = self._get_clipped_dist_mat(t1a_fam, t1a, t1a_indel)
                t2a_fam = self.fam[item2]
                t2a_indel = self.seq_indel[item2]
                t2a_dist_mat = self._get_clipped_dist_mat(t2a_fam, t2a, t2a_indel)

                t1_dist_mat = t1a_dist_mat[:t1_split, :t1_split]
                t2_dist_mat = t2a_dist_mat[t2_split:, t2_split:]

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

    def random_word(self, sentence):
        # print(sentence)
        tokens = list(sentence)
        output_label = []

        for i, token in enumerate(tokens):
            try:
                token_i = self.vocab[token]
            except KeyError:
                token_i = self.vocab['unk_index']

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
                    tokens[i] = token_i

                output_label.append(token_i)

            else:
                tokens[i] = token_i
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
    train_dataset = DatasetSeq('data/seq_unalign_indel_all_cut_head.csv', seq_len=256,
                               seq_mode='two', relative_3d=True)




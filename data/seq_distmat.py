import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
import os
import multiprocessing as mp
import matplotlib.pyplot as pl
import pickle
from numba import jit
import gzip
from time import time


def parse_pfam_align(pfam_file_path):
    with open(pfam_file_path, 'rt') as alignment:
        seqs = {}
        for line in alignment:
            line = line.strip()  # remove trailing \n
            if line == "":
                # blank line, ignore
                pass
            elif line[0].isalnum():
                # Sequence
                # Format: "<seqname> <sequence>"
                parts = [x.strip() for x in line.split(" ", 1)]
                if len(parts) != 2:
                    # This might be someone attempting to store a zero length sequence?
                    raise ValueError(
                        "Could not split line into identifier "
                        "and sequence:\n" + line)
                seq_id, seq = parts
                seqs.setdefault(seq_id, "")
                seqs[seq_id] += seq.replace(".", "")   # hmmer break long sequences up and interleave them
        return seqs


def seq_dist_mat_hmm_one(fam, hmm_length):
    # make a distance matrix for the hmm of this family
    dist_mat = np.zeros((hmm_length, hmm_length))
    dist_mat_count = np.zeros((hmm_length, hmm_length), dtype=int)
    pdb_dist_mat = pd.read_pickle(f'pfam_pdb_distmat/{fam}/pdb_distmat_{fam}.pkl.gz', compression='gzip')
    dist_mat_seq_list = pdb_dist_mat['dist_mat'].values
    # with open(f'pfam_pdb_distmat/{fam}/pdb_align_{fam}.txt', 'rt') as alignment:
    #     for line in alignment:
    #         if line == '':
    #             continue
    #         if line[0].isdigit():
    #             seq_num = int(line.split(' ')[0])
    #             seq = line[:-1].split(' ')[-1].replace('.', '')
    seqs = parse_pfam_align(f'pfam_pdb_distmat/{fam}/pdb_align_{fam}.txt')
    for seq_id, seq in seqs.items():
        dist_mat_seq = dist_mat_seq_list[int(seq_id)]
        # print(seq)
        # print(dist_mat_seq.shape)
        count_hmm_i = 0
        count_seq_i = 0
        for i, si in enumerate(seq):
            if si.islower():
                count_seq_i += 1
            elif si == '-':
                count_hmm_i += 1
            elif si.isupper():
                count_hmm_j = 0
                count_seq_j = 0
                for j, sj in enumerate(seq):
                    if sj.islower():
                        count_seq_j += 1
                    elif sj == '-':
                        count_hmm_j += 1
                    elif sj.isupper():
                        # print(count_seq_i, count_seq_j, count_hmm_i, count_hmm_j)
                        dist_mat[count_hmm_i, count_hmm_j] += dist_mat_seq[count_seq_i, count_seq_j]
                        dist_mat_count[count_hmm_i, count_hmm_j] += 1
                        count_hmm_j += 1
                        count_seq_j += 1
                count_hmm_i += 1
                count_seq_i += 1
    ind = (dist_mat_count == 0)
    dist_mat_count[ind] = 1
    dist_mat_hmm = dist_mat/dist_mat_count
    # dist_mat_hmm.to_pickle(f'pfam_pdb_distmat/{fam}/hmm_distmat_{fam}.pkl')
    with open(f'pfam_pdb_distmat/{fam}/hmm_distmat_{fam}.pkl', 'wb') as distmat_pkl:
        pickle.dump(dist_mat_hmm, distmat_pkl)


@jit(nopython=True)
def dist_mat_one(sindel, h, dist_mat, dist_mat_hmm):
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

# def dist_mat_one(seq, dist_mat, dist_mat_hmm):
#     count_hmm_i = 0
#     count_seq_i = 0
#     # print(seq)
#     for i, si in enumerate(seq):
#         if si.islower():
#             count_seq_i += 1
#         elif si == '-':
#             count_hmm_i += 1
#         elif si.isupper():
#             count_hmm_j = 0
#             count_seq_j = 0
#             for j, sj in enumerate(seq):
#                 if sj.islower():
#                     count_seq_j += 1
#                 elif sj == '-':
#                     count_hmm_j += 1
#                 elif sj.isupper():
#                     # print(count_seq_i, count_seq_j, count_hmm_i, count_hmm_j)
#                     dist_mat[count_seq_i, count_seq_j] = dist_mat_hmm[count_hmm_i, count_hmm_j]
#                     count_hmm_j += 1
#                     count_seq_j += 1
#             count_hmm_i += 1
#             count_seq_i += 1
#     return dist_mat


def seq_indel_one(fam):
    # save indel sequences of this family
    seq_id_list = []
    seq_unalign_list = []
    seq_indel_list = []
    seqs = parse_pfam_align(f'pfam_pdb_distmat/{fam}/pfam_full_align_{fam}.txt')
    for seq_id, seq in seqs.items():
        seq_id_list.append(seq_id)
        seq_unalign = seq.replace('-', '').upper()
        seq_unalign_list.append(seq_unalign)
        seq_indel_list.append(seq)

    seq_num = np.arange(len(seq_id_list))
    # print(fam, len(seq_num))
    df = pd.DataFrame({'seq_num': seq_num, 'seq_id': seq_id_list, 'seq_unalign': seq_unalign_list,
                       'seq_indel': seq_indel_list})
    df.to_csv(f'pfam_pdb_distmat/{fam}/pfam_seq_unalign_indel_{fam}.csv', index=False)
    with open('seq_distmat_log.txt', 'at') as slog:
        slog.write(f'{fam},{len(seq_num)}\n')

def seq_dist_mat_one(fam):
    # make distance matrices for all sequences in this family
    with open(f'pfam_pdb_distmat/{fam}/hmm_distmat_{fam}.pkl', 'rb') as distmat_pkl:
        dist_mat_hmm = pickle.load(distmat_pkl)
    seq_id_list = []
    seq_list = []
    dist_mat_list = []
    # with open(f'pfam_pdb_distmat/{fam}/pfam_full_align_{fam}.txt', 'rt') as alignment:
    #     for line in tqdm(alignment):
    #         if line == '':
    #             continue
    #         # if len(seq_list) > 1000:
    #         #     break
    #         if line[0].isalnum():
    #             seq_id = line.split(' ')[0]
    #             seq = line[:-1].split(' ')[-1].replace('.', '')
    seqs = parse_pfam_align(f'pfam_pdb_distmat/{fam}/pfam_full_align_{fam}.txt')
    for seq_id, seq in seqs.items():
        seq_id_list.append(seq_id)
        seq_unalign = seq.replace('-', '').upper()
        seq_list.append(seq_unalign)
        dist_mat = np.ones((len(seq_unalign), len(seq_unalign))) * (-1)
        sindel = np.zeros(len(seq), dtype=int)
        for i, si in enumerate(seq):
            if si.islower():
                sindel[i] = 1
            elif si == '-':
                sindel[i] = 2
        dist_mat = dist_mat_one(sindel, sindel.shape[0], dist_mat, dist_mat_hmm)
        # convert distmat into int grid values [-1, 0, 1, ..., 19, 20] * (2A)
        # dist_mat = (dist_mat // 2)
        dist_mat = np.round(dist_mat)
        dist_mat = np.clip(dist_mat, a_min=None, a_max=60).astype(np.int8)
        dist_mat_list.append(dist_mat)

    seq_num = np.arange(len(seq_id_list))
    # print(fam, len(seq_num))
    df = pd.DataFrame({'seq_num': seq_num, 'seq_id': seq_id_list, 'seq': seq_list})
    df.to_csv(f'pfam_pdb_distmat/{fam}/pfam_seq_unalign_{fam}.csv', index=False)
    # subprocess.run(['mkdir', f'pfam_pdb_distmat/{fam}/seq_distmat'])
    # for i, dist_mat in zip(seq_num, dist_mat_list):
    #     with open(f'pfam_pdb_distmat/{fam}/seq_distmat/seq_distmat_{fam}_{i}.pkl', 'wb') as distmat_pkl:
    #         pickle.dump(dist_mat, distmat_pkl)
    with open('seq_distmat_log.txt', 'at') as slog:
        slog.write(f'{fam},{len(seq_num)}\n')

    # df_dist_mat = pd.DataFrame({'dist_mat': dist_mat_list}, dtype=np.int8)
    # df_dist_mat.to_pickle(f'pfam_pdb_distmat/{fam}/seq_distmat_{fam}.pkl.gz', compression='gzip')
    # with gzip.open(f'pfam_pdb_distmat/{fam}/seq_distmat_{fam}.pkl.gz', 'wb') as f:
    #     f.write(pickle.dumps(dist_mat_list, pickle.HIGHEST_PROTOCOL))
    # dist_mat_list = pickle.load(gzip.open(f'pfam_pdb_distmat/{fam}/seq_distmat_{fam}.pkl.gz', 'rb'))


# make distance matrix for HMM profile
def seq_dist_mat_batch(pfam_acc_list):
    # print(pfam_acc_list)
    # for fam in pfam_acc_list[-1]:
    # seq_finished = pd.read_csv('seq_distmat_log.txt')['fam'].values
    # pfam_len = pd.read_csv('pfam_fam_len.csv')
    # pfam_len_dict = {x: y for x, y in zip(pfam_len['pfam_acc'], pfam_len['fam_len'])}

    for fam in pfam_acc_list:
        try:
            # with open(f'pfam_pdb_distmat/{fam}/pfam_full_align_{fam}.txt', 'rt') as alignment:
            #     i = 0
            #     for line in tqdm(alignment):
            #         i += 1
            # print(fam)
            # if fam not in seq_finished:
            #     # seq_dist_mat_one(fam)
            # hmm_length = pfam_len_dict[fam]
            seq_indel_one(fam)
        # except KeyError:
        except FileNotFoundError:
            print(f'No {fam} in Pfam-A')
    # print('done batch')


def seq_dist_mat_hmm_batch(pfam_acc_list):
    # print(pfam_acc_list)
    pfam_len = pd.read_csv('pfam_fam_len.csv')
    pfam_len_dict = {x: y for x, y in zip(pfam_len['pfam_acc'], pfam_len['fam_len'])}
    for fam in pfam_acc_list:
        try:
            hmm_length = pfam_len_dict[fam]
            seq_dist_mat_hmm_one(fam, hmm_length)
        except KeyError:
            print(f'No {fam} in Pfam-A')
    print('done batch')


def seq_dist_mat_all():
    pfam_acc_list = pd.read_csv('pfam_acc_list.txt')['pfam_acc'].values
    np.random.shuffle(pfam_acc_list)
    num = pfam_acc_list.shape[0]
    batch_size = 160

    batch_list = []
    for i in range(0, num, batch_size):
        batch = pfam_acc_list[i:i+batch_size]
        batch_list += [batch]

    # setup the multi-processes
    num_cores = 40
    with mp.Pool(processes=num_cores) as pool:
        # pool.map(seq_dist_mat_hmm_batch, batch_list)
        pool.map(seq_dist_mat_batch, batch_list)


def del_dist_mat():
    pfam_acc_list = pd.read_csv('pfam_acc_list.txt')['pfam_acc'].values
    with open('del_distmat.sh', 'wt') as dfile:
        for fam in pfam_acc_list:
            dfile.write(f'rm -rf pfam_pdb_distmat/{fam}/seq_distmat\n')
            dfile.write(f'echo {fam} >> del_distmat_log.txt\n')


def test_dist_mat_one():
    # 1 - gap, 2 - del
    # 012.3456.7
    # ABC.DEFG.H   # hmm
    # A-CxDE-GxH
    # 0-2x34-6x7
    # 02x346x7
    # sindel = np.array([0, 2, 0, 1, 0, 0, 2, 0, 1, 0])
    # dist_mat_hmm = np.arange(64).reshape(8, 8)
    # dist_mat = np.zeros((8, 8))

    # 0123.456.7
    # ABCD.EFG.H  # hmm
    # A---xE-GxH
    # 0---x4-6x7
    # 0x46x7
    sindel = np.array([0, 2, 2, 2, 1, 0, 2, 0, 1, 0])
    dist_mat_hmm = np.arange(64).reshape(8, 8)
    dist_mat = np.zeros((6, 6))

    dist_mat = dist_mat_one(sindel, sindel.shape[0], dist_mat, dist_mat_hmm)
    pl.figure()
    pl.imshow(dist_mat)
    pl.figure()
    pl.imshow(dist_mat_hmm)


if __name__ == '__main__':
    seq_dist_mat_all()
    # seqs = parse_pfam_align('pf00001.stk')
    # t0 = time()
    # seq_dist_mat_one('PF00001')
    # print(time() - t0)
    # test_dist_mat_one()
    # seq_indel_one('PF00001')


# df = pd.read_csv('seq_distmat_log.txt')
# with open('del.sh', 'wt') as d:
#     for fam in df['pf']:
#         d.write(f'rm -f pfam_pdb_distmat/{fam}/seq_distmat_{fam}.pkl.gz\n')

# if os.path.exists(f'pfam_pdb_distmat/{fam}/seq_distmat_{fam}.pkl.gz'):
#     subprocess.run(['rm'])


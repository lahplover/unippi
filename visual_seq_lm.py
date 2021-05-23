import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DatasetSeqLM
from model import SeqLM, ProLM
from trainer import SeqLMTrainer
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import pandas as pd


# visualize seq_lm_gpt model
# restart_file = 'checkpoints/seq_lm_a4_ep0'
#
# print("Building BERT model")
# # Initialize the BERT Language Model, with BERT model
# model = SeqLM(23, hidden=1024, n_layers=8, attn_heads=8)
#
# print("reload pretrained BERT model")
# model.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))
#
# print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
#
# for param in model.embedding.token.parameters():
#     eb_vec = param


# visualize pro_lm model
restart_file = 'checkpoints/prolm1_ep0'
model = ProLM(23, hidden=512, n_layers=5, attn_heads=1)

print("reload pretrained BERT model")
model.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))

for param in model.prolm.token_embedding.parameters():
    eb_vec = param



x = eb_vec.detach().numpy()

# xn = np.zeros((20, 1024))
# for i in range(x.shape[1]):
#     a = x[1:21, i]
#     xn[:, i] = (a - a.mean()) / np.sqrt(a.var())
#

pca = PCA(n_components=2)
xr = pca.fit(x).transform(x)

amino_acids = pd.read_csv('data/amino_acids.csv')
vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
vocab['pad_index'] = 0
vocab['start_index'] = 21
vocab['unk_index'] = 22

idx2vocab = {y: x for x, y in vocab.items()}

df = pd.DataFrame({})

pl.figure()
for i in range(0, 23, 1):
    # pl.scatter(xr[i, 0], xr[i, 1], color='green')
    pl.scatter(xr[i, 0], xr[i, 1])
    pl.text(xr[i, 0], xr[i, 1], f'{idx2vocab[i]}')

    df[f'{idx2vocab[i]}'] = x[i]



pl.figure()
for i in range(x.shape[0]):
    pl.subplot(23, 1, i+1)
    pl.plot(x[i])
    pl.ylim(-1, 1)





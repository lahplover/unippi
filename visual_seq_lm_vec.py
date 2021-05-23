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
from tqdm import tqdm

device = 'cpu'

# visualize seq_lm_gpt model
restart_file = 'checkpoints/seq_lm_a4_ep0'

print("Building BERT model")
# Initialize the BERT Language Model, with BERT model
model = SeqLM(23, hidden=1024, n_layers=8, attn_heads=8)

print("reload pretrained BERT model")
model.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))

print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

model.eval()
torch.set_grad_enabled(False)


test_dataset = DatasetSeqLM('data/aa_simple.csv', seq_len=5)
test_data_loader = DataLoader(test_dataset, batch_size=10)


for i, data in tqdm(enumerate(test_data_loader)):
    data = {key: value.to(device) for key, value in data.items()}
    x = data["seq_x"]
    padding_mask = (x == 0)
    x_emb = model.embedding(x)
    x_vec = model.encoder(x_emb, padding_mask)
    x_vec = x_vec.transpose(0, 1)

    break








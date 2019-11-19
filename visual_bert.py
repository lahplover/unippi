import os
import numpy as np
import torch
import torch.nn as nn
from data import DatasetSeq, DatasetInterDomain, DatasetPDB
from model import BERTLM, BERT
from trainer import BERTTrainer
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as pl


def load_dataset(args):
    if args.task == 'pfam':
        train_data_path = args.train_dataset
        print("Loading Train Dataset", train_data_path)
        train_dataset = DatasetSeq(train_data_path, seq_len=args.seq_len, seq_mode=args.seq_mode,
                                   relative_3d=args.relative_3d,
                                   relative_3d_size=10, relative_3d_step=2)
    else:
        raise ValueError('unknown task name')

    return train_dataset


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)


# GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes
gpu_mode = 1
if not torch.cuda.is_available():
    gpu_mode = 0

if (gpu_mode == 1) & torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


train_dataset = load_dataset(args)

print("Creating Dataloader")

# datasampler = RandomSampler(train_dataset)
# train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#                                num_workers=args.num_workers, sampler=datasampler)

# build model
print("Building BERT model")
model = BERT(len(train_dataset.vocab),
             hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,
             seq_mode=args.seq_mode,
             abs_position_embed=args.abs_position_embed,
             relative_attn=args.relative_attn,
             relative_1d=args.relative_1d,
             max_relative_1d_positions=10,
             relative_3d=args.relative_3d,
             relative_3d_vocab_size=len(train_dataset.vocab_3d),
             visual=True)

if args.restart:
    print("reload pretrained BERT model")
    model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')))

model.to(device)

torch.set_grad_enabled(False)

print(model.visual)

item = 2
data = train_dataset[item]

data = {key: value.unsqueeze(0).to(device) for key, value in data.items()}

x, output = model.forward(data["bert_input"], distance_matrix=data["dist_mat"])


for param in model.embedding.token.parameters():
    eb_vec = param

for att in output['attention']:
    pl.figure()
    pl.imshow(att[0].numpy()[2])
    pl.colorbar()

pl.figure()
pl.imshow(x[0].numpy())
pl.colorbar()



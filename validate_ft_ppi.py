import os
import numpy as np
import torch
import torch.nn as nn
from model import FTPPI, FTPPIOneHot
from trainer import FTPPITrainer
from data import DatasetFTInterFam
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter


def load_dataset(args):
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = DatasetFTInterFam(args.test_dataset, seq_len=args.seq_len,
                                         relative_3d=args.relative_3d,
                                         relative_3d_size=10, relative_3d_step=2,
                                         target_intra_dm=args.target_intra_dm)
    return test_dataset


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)

# GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes
gpu_mode = 1
cuda_device_id = 0

# Setup cuda device
if (gpu_mode == 1) & torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


test_dataset = load_dataset(args)

test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# build model
print("Building model")
if not args.ft_ppi_no_pretrain:
    model = FTPPI(len(test_dataset.vocab),
                    hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,
                    abs_position_embed=args.abs_position_embed,
                    relative_attn=args.relative_attn,
                    relative_1d=args.relative_1d,
                    max_relative_1d_positions=10,
                    relative_3d=args.relative_3d,
                    relative_3d_vocab_size=len(test_dataset.vocab_3d))
else:
    model = FTPPIOneHot(len(test_dataset.vocab), args.hidden)

print("reload pretrained model")
model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')),
                           strict=False)

model.to(device)

if (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS" % torch.cuda.device_count())
    model = nn.DataParallel(model)

# print(f'exp_{args.exp_i}, learning_rate: {args.lr}, weight_decay: {args.weight_decay}')
writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

trainer = FTPPITrainer(writer, model,
                    train_dataloader=test_data_loader,
                    test_dataloader=test_data_loader,
                    lr=args.lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    warmup_steps=args.warmup_steps,
                    lr_scheduler=args.lr_scheduler,
                    device=device,
                    log_freq=args.log_freq)

trainer.test(1)



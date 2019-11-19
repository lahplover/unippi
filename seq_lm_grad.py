import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DatasetSeqLM
from model import SeqLM
from trainer import SeqLMTrainer
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as pl


def load_dataset(args):
    train_data_path = args.train_dataset
    print("Loading Train Dataset", train_data_path)
    train_dataset = DatasetSeqLM(train_data_path, seq_len=args.seq_len)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = DatasetSeqLM(train_data_path, seq_len=args.seq_len) \
        if args.test_dataset is not None else None

    return train_dataset, test_dataset


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)

if args.multi_node:
    distributed.init_process_group(
        backend=args.backend,
        init_method='env://'
        # world_size=args.world_size,
        # rank=args.rank,
    )

# GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes
gpu_mode = 1
cuda_device_id = 0
if distributed.is_available():
    if distributed.is_initialized():
        gpu_mode = 2
        cuda_device_id = args.local_rank
elif not torch.cuda.is_available():
    gpu_mode = 0

# Setup cuda device
if gpu_mode == 2:
    torch.cuda.set_device(cuda_device_id)
    device = torch.device(f"cuda:{cuda_device_id}")
elif (gpu_mode == 1) & torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# print(f"world size: {distributed.get_world_size()}")
# print(f"rank: {distributed.get_rank()}")
# for large dataset, prepare a subset of dataset for each GPU, load the dataset with id=get_rank()
# print("Loading Train Dataset", args.train_dataset)
# train_data_path = args.train_dataset

train_dataset, test_dataset = load_dataset(args)

# print("Creating Dataloader")
# if gpu_mode == 2:
#     datasampler = DistributedSampler(train_dataset, shuffle=True)
#     # datasampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
#     train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#                                    num_workers=args.num_workers, sampler=datasampler)
#     # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#     #                                num_workers=args.num_workers)
# else:
#     datasampler = RandomSampler(train_dataset)
#     train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#                                    num_workers=args.num_workers, sampler=datasampler)
#     # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#     #                                num_workers=args.num_workers)
#
# test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
#     if test_dataset is not None else None

# build model
print("Building BERT model")
# Initialize the BERT Language Model, with BERT model
model = SeqLM(len(train_dataset.vocab),
              hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

if args.restart:
    print("reload pretrained BERT model")
    model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')))

model.to(device)

# Distributed GPU training if CUDA can detect more than 1 GPU
if gpu_mode == 2:
    print(f"Using GPU {cuda_device_id}")
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cuda_device_id],
                                                output_device=cuda_device_id,
                                                find_unused_parameters=True)
elif (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model)

# model.eval()


def get_score(item):
    data = train_dataset[item]

    data = {key: value.unsqueeze(dim=0).to(device) for key, value in data.items()}

    x, y = data['seq_x'], data['seq']  # (N, S)
    # print(x.size())
    #
    seq_len = x.size(1)
    # x = x.repeat((seq_len, 1))
    # x = torch.tril(x)
    #
    # y_tgt = torch.zeros((seq_len, seq_len), dtype=torch.long, device=x.device, requires_grad=False)
    # for i in range(seq_len):
    #     y_tgt[i, i] = y[i]

    padding_mask = (x == 0)

    x = model.embedding(x)

    x.requires_grad_()
    x.retain_grad()

    x_vec = model.encoder(x, padding_mask)

    # print('in model output size ', x.size())
    # return self.next_sentence(x), self.mask_lm(x)
    y_pred = model.seq_lm(x_vec)  # (N, S, E)
    # y_pred = y_pred.transpose(1, 2)

    # loss_fn = torch.nn.NLLLoss(ignore_index=0, reduction='none')
    # loss = loss_fn(y_pred, y)

    # loss_fn = torch.nn.NLLLoss(ignore_index=0)

    score = []
    for i in range(seq_len):
        # loss = loss_fn(y_pred[:, :, i], y[:, i])
        # g = torch.zeros(seq_len).unsqueeze(0)
        # g[0, i] = 1.0
        # loss.backward(retain_graph=True)
        # loss.backward(gradient=g, retain_graph=True)
        k = y[0, i]
        if k == 0:
            break
        y_pred[0, i, k].backward(retain_graph=True)
        x_grad = x.grad
        # x_grad = x_grad / loss[0, i].item()
        x_score = ((x_grad / x.detach()) ** 2).sum(dim=-1)
        # x_score[x_score == 0] = 1e-9
        x_score = x_score / x_score.sum(dim=1) * (1.0*i/seq_len)
        score.append(x_score)
        x.grad.zero_()

    score = np.vstack(score)
    pl.figure()
    pl.imshow(score)
    pl.colorbar()


for i in range(0, 100, 10):
    get_score(i)


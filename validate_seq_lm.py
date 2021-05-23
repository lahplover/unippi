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
# import matplotlib.pyplot as pl
from tqdm import tqdm
import pandas as pd


def load_dataset(args):
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = DatasetSeqLM(args.test_dataset, seq_len=args.seq_len)
    return test_dataset


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

test_dataset = load_dataset(args)

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

# datasampler = RandomSampler(test_dataset)
# test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
#                               sampler=datasampler)

test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# build model
print("Building BERT model")
# Initialize the BERT Language Model, with BERT model
# model = SeqLM(len(test_dataset.vocab),
#               hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
model = ProLM(len(test_dataset.vocab),
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

model.eval()
torch.set_grad_enabled(False)

writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

trainer = SeqLMTrainer(writer, model,
                      train_dataloader=test_data_loader,
                      test_dataloader=test_data_loader,
                      lr=args.lr,
                      betas=(args.adam_beta1, args.adam_beta2),
                      weight_decay=args.adam_weight_decay,
                      warmup_steps=args.warmup_steps,
                      lr_scheduler=args.lr_scheduler,
                      device=device,
                      log_freq=args.log_freq)

# print('Testing')
# trainer.test(1)
#
amino_acids = pd.read_csv('data/amino_acids.csv')
vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.idx)}
vocab['pad_index'] = 0
vocab['start_index'] = 21
vocab['unk_index'] = 22

idx2vocab = {y: x for x, y in vocab.items()}

avg_loss = 0.0
total_correct = 0
total_element = 0
len_data_loader = len(test_data_loader)

loss_fn = nn.NLLLoss(ignore_index=0)

for i, data in tqdm(enumerate(test_data_loader)):
    data = {key: value.to(device) for key, value in data.items()}

    seq_output = model.forward(data["seq_x"])

    # NLLLoss of predicting masked token word
    # (N, T, E) --> (N, E, T)
    # print(seq_output.transpose(1, 2).argmax(dim=1))
    loss = loss_fn(seq_output.transpose(1, 2), data["seq"])

    # masked token prediction accuracy
    idx = (data["seq"] > 0)
    # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx])
    # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]))
    correct = seq_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["seq"][idx]).sum().item()
    batch_n_element = data["seq"][idx].nelement()

    print(f'loss: {loss}  Accuracy/test: {100.0 * correct / batch_n_element}')

    avg_loss += loss.item()

    total_correct += correct
    total_element += batch_n_element

    with open('data/seq_lm_prediction.txt', 'at') as sf:
        seq_output = seq_output.detach().cpu().transpose(1, 2).argmax(dim=1).numpy()  # (N, T)
        seq_target = data['seq'].cpu().numpy()
        for j in range(seq_output.shape[0]):
            ind = (seq_target[j] > 0)
            s_pred = ''.join([idx2vocab[x] for x in seq_output[j][ind]])
            s_target = ''.join([idx2vocab[x] for x in seq_target[j][ind]])
            print(s_pred, s_target)

            sf.write(s_target + ',' + s_pred + '\n')

    # if i >= 1:
    #     break

print(f"avg_loss= {avg_loss / len_data_loader}, total_acc= {total_correct * 100.0 / total_element}")



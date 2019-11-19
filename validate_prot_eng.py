import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import FTProtEng
from trainer import FTProtEngTrainer
from data import DatasetProtEng
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_dataset(args):
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = DatasetProtEng(args.test_dataset, seq_len=args.seq_len,
                                  relative_3d=args.relative_3d,
                                  relative_3d_size=10, relative_3d_step=2,
                                  regression=args.regression
                                  )
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
model = FTProtEng(len(test_dataset.vocab),
                hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,
                abs_position_embed=args.abs_position_embed,
                relative_attn=args.relative_attn,
                relative_1d=args.relative_1d,
                max_relative_1d_positions=10,
                relative_3d=args.relative_3d,
                relative_3d_vocab_size=len(test_dataset.vocab_3d),
                regression=args.regression)

print("reload pretrained model")
model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')),
                           strict=False)

model.to(device)

if (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS" % torch.cuda.device_count())
    model = nn.DataParallel(model)


torch.set_grad_enabled(False)

avg_loss = 0
flux_target = np.array([])
flux_pred = np.array([])
for i, data in tqdm(enumerate(test_data_loader)):
    flux_target = np.append(flux_target, data["flux"].numpy())

    # 0. batch_data will be sent into the device(GPU or cpu)
    data = {key: value.to(device) for key, value in data.items()}

    flux_pred_i = model(data["seq_input"], distance_matrix=data["dist_mat"])

    if args.regression:
        flux_pred = np.append(flux_pred, flux_pred_i.detach().cpu().numpy())
        loss_fn = nn.MSELoss()
    else:
        flux_pred = np.append(flux_pred, flux_pred_i.argmax(dim=-1).detach().cpu().numpy())
        cls_weight = torch.tensor([10, 1], device=device, requires_grad=False, dtype=torch.float)
        loss_fn = nn.NLLLoss(weight=cls_weight)

    loss = loss_fn(flux_pred_i, data["flux"])

    avg_loss += loss.item()

print(avg_loss / len(test_data_loader))

df = pd.DataFrame({'flux_target': flux_target, 'flux_pred': flux_pred})
df.to_csv(args.test_dataset[:-4] + '_pred.csv', index=False)


if args.visual:
    import matplotlib.pyplot as pl
    from scipy.stats import spearmanr

    df = pd.read_csv('fluorescence_train_pred.csv')
    print('train: ', spearmanr(df['flux_target'], df['flux_pred']))
    pl.figure()
    pl.plot(df['flux_target'], df['flux_pred'], 'b.')

    df = pd.read_csv('fluorescence_test_pred.csv')
    print('test: ', spearmanr(df['flux_target'], df['flux_pred']))

    pl.figure()
    pl.plot(df['flux_target'], df['flux_pred'], 'g.')

    p = df['flux_pred'].values
    t = df['flux_target'].values

    print(p[(p==1) & (t==1)].shape,
        p[(p == 0) & (t == 1)].shape,
        p[(p==1) & (t==0)].shape,
        p[(p == 0) & (t == 0)].shape)



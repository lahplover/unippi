import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from optim_schedule import ScheduledOptim
from tqdm import tqdm
import pandas as pd
import numpy as np


class Seq2SeqTrainer:
    def __init__(self, writer, model,
                 train_dataloader, test_dataloader=None,
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 warmup_steps=1000,
                 lr_scheduler='decay',
                 device='cpu',
                 log_freq=10,
                 label_smoothing=True):
        """
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param log_freq: logging frequency of the batch iteration
        """

        self.model = model
        self.device = device

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler == 'cycle':
            self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=3e-5, max_lr=1e-3,
                                                                    step_size_up=100, cycle_momentum=False)
        else:
            self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps, init_lr=lr)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.label_smoothing = label_smoothing
        if label_smoothing:
            print('applying label smoothing')
            self.smooth_prob = 0.5
            self.criterion = nn.KLDivLoss(reduction='sum')
            df = pd.read_csv('data/blosum_prob.csv', dtype=np.float32)
            # vocab = 23, pad_index = 0, 2 other indexes
            submat = np.eye(20, dtype=np.float32) * self.smooth_prob + (1 - self.smooth_prob) * df.values
            submat = np.pad(submat, ((1, 2), (1, 2)), 'constant', constant_values=0)
            self.submat = torch.tensor(submat, device=self.device, requires_grad=False)
        else:
            # self.criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
            self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.writer = writer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, output_path=None):
        self.iteration(epoch, self.train_data, train=True, output_path=output_path)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    # def _smooth_label(self, y):
    #     """
    #     smooth label based on the BLOSUM62 matrix
    #     :param y: (N, T)
    #     :return ys: (N, T, E)
    #     """
    #
    #     ys = self.submat[y]
    #
    #     return ys

    def iteration(self, epoch, data_loader, train=True, output_path=None):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        len_data_loader = len(data_loader)

        for i, data in tqdm(enumerate(data_loader)):
            data = {key: value.to(self.device) for key, value in data.items()}

            seq_output = self.model.forward(data["src"], data["tgt_x"])

            if self.label_smoothing:
                # ys = self._smooth_label(data["tgt_y"])
                ys = self.submat[data["tgt_y"]]
                loss = self.criterion(seq_output, ys)
            else:
                # NLLLoss of predicting masked token word
                # (N, T, E) --> (N, E, T)
                # print(seq_output.transpose(1, 2))
                loss = self.criterion(seq_output.transpose(1, 2), data["tgt_y"])

            # 3. backward and optimization only in train
            if train:
                if self.lr_scheduler == 'cycle':
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.optim_schedule.step()
                else:
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

            # masked token prediction accuracy
            idx = (data["tgt_y"] > 0)
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx])
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]))
            correct = seq_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["tgt_y"][idx]).sum().item()
            batch_n_element = data["tgt_y"][idx].nelement()
            total_correct += correct
            total_element += batch_n_element

            avg_loss += loss.item()

            if train:
                # print("write train loss")
                self.writer.add_scalar('Loss/train', loss, epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy/train', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)
            else:
                self.writer.add_scalar('Loss/test', loss, epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy/test', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)

            save_step = 10000
            if output_path is not None:
                if i % save_step == 0:
                    output_path_step = output_path + f'_st{i // save_step}'
                    torch.save(self.model.module.state_dict(), output_path_step)
                    print(f"EP:{epoch} Step{i // save_step} Model Saved on {output_path_step}")

            # torch.cuda.empty_cache()

        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader, "total_acc=",
              total_correct * 100.0 / total_element)


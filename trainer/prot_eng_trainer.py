import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from optim_schedule import ScheduledOptim
from tqdm import tqdm


class FTProtEngTrainer:
    """
    """

    def __init__(self, writer, model,
                 train_dataloader, test_dataloader=None,
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 warmup_steps=1000,
                 lr_scheduler='decay',
                 device='cpu',
                 log_freq=10,
                 regression=True):
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
        self.regression = regression

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

        if self.regression:
            self.loss = nn.MSELoss()
            # self.loss = nn.L1Loss()
        else:
            cls_weight = torch.tensor([10, 1], device=device, requires_grad=False, dtype=torch.float)
            self.loss = nn.NLLLoss(weight=cls_weight)

        self.log_freq = log_freq
        self.writer = writer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration_two(epoch, self.train_data)

    def test(self, epoch):
        # disable gradients to save memory
        # torch.set_grad_enabled(False)
        self.iteration_two(epoch, self.test_data, train=False)

    def iteration_two(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_loss = 0.0
        len_data_loader = len(data_loader)

        if not self.regression:
            total_correct = 0
            total_element = 0

        for i, data in tqdm(enumerate(data_loader)):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output = self.model(data["seq_input"],
                                          distance_matrix=data["dist_mat"])

            loss = self.loss(next_sent_output, data["flux"])

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

            avg_loss += loss.item()

            # prediction accuracy
            if not self.regression:
                correct = next_sent_output.argmax(dim=-1).eq(data["flux"]).sum().item()
                batch_n_element = data["flux"].nelement()
                total_correct += correct
                total_element += batch_n_element

            if train:
                # print("write train loss")
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len_data_loader + i)
                if not self.regression:
                    self.writer.add_scalar('Accuracy/train', 100.0 * correct / batch_n_element,
                                           epoch * len_data_loader + i)
            else:
                self.writer.add_scalar('Loss/test', loss.item(), epoch * len_data_loader + i)
                if not self.regression:
                    self.writer.add_scalar('Accuracy/test', 100.0 * correct / batch_n_element,
                                           epoch * len_data_loader + i)

        if self.regression:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader)
        else:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader, "total_acc=",
                  total_correct * 100.0 / total_element)


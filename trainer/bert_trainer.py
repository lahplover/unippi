import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from optim_schedule import ScheduledOptim
from tqdm import tqdm

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, writer, model, seq_mode,
                 train_dataloader, test_dataloader=None,
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 warmup_steps=1000,
                 lr_scheduler='decay',
                 device='cpu',
                 log_freq=10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        self.model = model
        self.device = device

        self.seq_mode = seq_mode

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
        self.loss_masked = nn.NLLLoss(ignore_index=0)
        self.loss_next = nn.NLLLoss()

        self.log_freq = log_freq
        self.writer = writer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        if self.seq_mode == 'two':
            self.iteration_two(epoch, self.train_data)
        elif self.seq_mode == 'one':
            self.iteration_one(epoch, self.train_data)

    def test(self, epoch):
        # disable gradients to save memory
        torch.set_grad_enabled(False)

        if self.seq_mode == 'two':
            self.iteration_two(epoch, self.test_data, train=False)
        elif self.seq_mode == 'one':
            self.iteration_one(epoch, self.test_data, train=False)

    def iteration_two(self, epoch, data_loader, train=True):
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
        total_correct_2 = 0
        total_element_2 = 0
        len_data_loader = len(data_loader)

        for i, data in tqdm(enumerate(data_loader)):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"],
                                                                  distance_matrix=data["dist_mat"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.loss_next(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.loss_masked(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss
            # loss = next_loss

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
            idx = (data["bert_label"] > 0)
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx])
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]))
            correct = mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]).sum().item()
            batch_n_element = data["bert_label"][idx].nelement()
            total_correct += correct
            total_element += batch_n_element
            # print(correct, data["bert_label"][idx].nelement())

            # next sentence prediction accuracy
            correct2 = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            batch_n_element2 = data["is_next"].nelement()
            total_correct_2 += correct2
            total_element_2 += batch_n_element2

            avg_loss += loss.item()

            if train:
                # print("write train loss")
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy1/train', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy2/train', 100.0 * correct2 / batch_n_element2, epoch*len_data_loader + i)

            else:
                self.writer.add_scalar('Loss/test', loss.item(), epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy1/test', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy2/test', 100.0 * correct2 / batch_n_element2, epoch*len_data_loader + i)

            # print(i, loss)
            # self.writer.add_scalar('Loss', loss, epoch*len_data_loader + i)
            # self.writer.add_scalar('Accuracy', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)

        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader,
              "total_acc1=", total_correct * 100.0 / total_element,
              "total_acc2=", total_correct_2 * 100.0 / total_element_2)

    def iteration_one(self, epoch, data_loader, train=True):
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
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            # next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            mask_lm_output = self.model.forward(data["bert_input"], distance_matrix=data["dist_mat"])

            # 2. NLLLoss of predicting masked token word
            loss = self.loss_masked(mask_lm_output.transpose(1, 2), data["bert_label"])

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
            idx = (data["bert_label"] > 0)
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx])
            # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]))
            correct = mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]).sum().item()
            batch_n_element = data["bert_label"][idx].nelement()
            total_correct += correct
            total_element += batch_n_element
            # print(correct, data["bert_label"][idx].nelement())

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            avg_loss += loss.item()

            if train:
                # print("write train loss")
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy/train', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)
            else:
                self.writer.add_scalar('Loss/test', loss.item(), epoch*len_data_loader + i)
                self.writer.add_scalar('Accuracy/test', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)
            # print(i, loss)
            # self.writer.add_scalar('Loss', loss, epoch*len_data_loader + i)
            # self.writer.add_scalar('Accuracy', 100.0 * correct / batch_n_element, epoch*len_data_loader + i)

        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len_data_loader, "total_acc=",
              total_correct * 100.0 / total_element)


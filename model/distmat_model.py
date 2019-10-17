import torch.nn as nn
from .bert import BERT


class InterDM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8,
                 dm_hidden=16,
                 seq_mode='two',
                 abs_position_embed=True,
                 relative_attn=False,
                 relative_1d=False,
                 max_relative_1d_positions=10,
                 relative_3d=False,
                 relative_3d_vocab_size=10):

        super().__init__()
        self.seq_mode = seq_mode

        self.bert = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,
                         seq_mode=seq_mode,
                         abs_position_embed=abs_position_embed,
                         relative_attn=relative_attn,
                         relative_1d=relative_1d,
                         max_relative_1d_positions=max_relative_1d_positions,
                         relative_3d=relative_3d,
                         relative_3d_vocab_size=relative_3d_vocab_size)

        self.dist_mat_pred = DistMatPrediction(hidden, dm_hidden, relative_3d_vocab_size)

    def forward(self, x, segment_label=None, distance_matrix=None):
        x = self.bert(x, segment_label, distance_matrix)
        # dist_mat = self._get_dist_mat(x, segment_label)
        dist_mat = self.dist_mat_pred(x)
        return dist_mat

    # def _get_dist_mat(self, x, segment_label):
    #     """
    #     use the angular alignment of two vectors as their distance?
    #     use (x1-x2) as distance?
    #     """
    #     t1 = segment_label[segment_label == 1].size(0)
    #     t2 = segment_label[segment_label == 2].size(0)
    #     x1 = x[:, 1:t1-1]  # exclude the padded sos_index and eos_index
    #     x2 = x[:, t1:t1+t2-1]  # exclude the padded eos_index
    #     x1 = x1[None, :, :, :]
    #     x2 = x2[:, None, :, :]
    #     x12 = (x1 * x2).sum(dim=3)
    #     dist_mat = x12  #
    #     return dist_mat


class DistMatPrediction(nn.Module):
    """
    # map the seq vector to distance
    """
    def __init__(self, hidden, dm_hidden, relative_3d_vocab_size):
        super().__init__()
        # exclude sos_index, eos_index, pad_index, inter_12 from the vocab
        d_size = relative_3d_vocab_size-4   # number of distance class + no_msa
        self.linear1 = nn.Linear(hidden, dm_hidden)
        self.linear2 = nn.Linear(dm_hidden, d_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        :param x:  shape (N, S, E)
        :return: dist_mat, shape (N, d_size, S, S)
        """
        # t1 = segment_label[segment_label == 1].size(0)
        # t2 = segment_label[segment_label == 2].size(0)
        # x1 = x[:, 1:t1-1]  # exclude the padded sos_index and eos_index
        # x2 = x[:, t1:t1+t2-1]  # exclude the padded eos_index
        # x1 = x1[:, :, None, :]   # shape (N, t1-2, 1, E)
        # x2 = x2[:, None, :, :]   # shape (N, 1, t2-1, E)

        x = self.linear1(x)
        x1 = x[:, :, None, :]   # shape (N, S, 1, E)
        x2 = x[:, None, :, :]   # shape (N, 1, S, E)

        dist_mat = self.softmax(self.linear2((x1-x2)**2))  # (N, S, S, d_size)
        dist_mat = dist_mat.transpose(1, 3)  # (N, d_size, S, S)
        return dist_mat


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


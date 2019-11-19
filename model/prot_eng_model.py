import torch.nn as nn
import torch
from .bert import BERT


class FTProtEng(nn.Module):
    """
    finetune the Next Sentence Prediction Model
    """

    def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8,
                 seq_mode='one',
                 abs_position_embed=True,
                 relative_attn=False,
                 relative_1d=False,
                 max_relative_1d_positions=10,
                 relative_3d=False,
                 relative_3d_vocab_size=10,
                 regression=True):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

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

        self.next_sentence = NextSentencePrediction(hidden, regression=regression)

    def forward(self, x, segment_label=None, distance_matrix=None):
        # x (N, S)
        mask_pad = (x > 0).unsqueeze(-1)
        x = self.bert(x, segment_label, distance_matrix)
        return self.next_sentence(x, mask_pad)


# class NextSentencePrediction(nn.Module):
#     """
#     2-class classification model : is_next, is_not_next
#     x: (N, S, E) -> (N, E) -> (N, 2)
#     """
#
#     def __init__(self, hidden):
#         """
#         :param hidden: BERT model output size
#         """
#         super().__init__()
#         # self.compute_weight = nn.Linear(hidden, 1)
#         # self.norm1 = nn.LayerNorm(hidden)
#         # self.norm2 = nn.LayerNorm(hidden)
#         self.linear1 = nn.Linear(hidden, hidden)
#         # self.linear2 = nn.Linear(hidden, 1)
#         self.linear2 = nn.Linear(hidden, 5)
#         self.activation = nn.ReLU()
#         self.softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, x, mask_pad):
#         # attention_weight = self.compute_weight(self.norm1(x))  # (N, S, E) -> (N, S, 1)
#         # attention_weight = attention_weight.masked_fill(mask_pad == 0, -1e9)
#         # attention = nn.functional.softmax(attention_weight, dim=1)
#         # # print(x.size(), attention.size())
#         # x = torch.matmul(attention.transpose(1, 2), x).squeeze()  # (N, 1, S) * (N, S, E)
#         # # print(x.size())
#         # x = self.norm2(x)
#         x = x.mean(dim=1).squeeze()
#         x = self.activation(self.linear1(x))
#         # return self.linear2(x).squeeze()
#         return self.softmax(self.linear2(x))  # (N, 2)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    x: (N, S, E) -> (N, E) -> (N, 2)
    """

    def __init__(self, hidden, regression=True):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.compute_weight = nn.Linear(hidden, 1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.linear1 = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()

        self.regression = regression
        if self.regression:
            self.linear2 = nn.Linear(hidden, 1)
        else:
            self.linear2 = nn.Linear(hidden, 2)
            self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, mask_pad):
        attention_weight = self.compute_weight(self.norm1(x))  # (N, S, E) -> (N, S, 1)
        attention_weight = attention_weight.masked_fill(mask_pad == 0, -1e9)
        attention = nn.functional.softmax(attention_weight, dim=1)
        # print(x.size(), attention.size())
        x = torch.matmul(attention.transpose(1, 2), x).squeeze()  # (N, 1, S) * (N, S, E)
        # print(x.size())
        x = self.norm2(x)
        # x = x.mean(dim=1).squeeze()
        x = self.activation(self.linear1(x))
        if self.regression:
            return self.linear2(x).squeeze()
        else:
            return self.softmax(self.linear2(x))  # (N, 2)


import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RelativeAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, num_heads, d_model, d_relation=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        self.d_relation = d_relation

        if self.d_relation is not None:
            self.relation_linear = nn.Linear(self.d_k, d_relation)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_bias_matrix, mask=None, output=None):
        """
        forward function for RelativeAttention.
        :param query: [N, S, E]
        :param key:  [N, S, E]
        :param value: [N, S, E]
        :param mask: Optional
        :param relation_bias_matrix: [N, S, S, dk]
        :return:
        """
        batch_size, seq_len, d_model = query.size()

        # 1) Do all the linear projections in batch, reshape from d_model => num_heads x d_k
        # query, key, value have shapes [N, num_heads, S, dk]
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # first attention term, [N, num_heads, S, dk] * [N, num_heads, dk, S] -> [N, num_heads, S, S]
        score1 = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # second attention term (relative attention)
        if self.d_relation is None:
            # N * S parallel multiplications of [num_heads, dk] and [dk, S] matrices
            # [N, S, num_heads, dk] * [N, S, dk, S] -> [N, S, num_heads, S], transpose to [N, num_heads, S, S]
            score2 = torch.matmul(query.transpose(1, 2), relation_bias_matrix.transpose(-2, -1)).transpose(1, 2)
        else:
            assert(self.d_relation == relation_bias_matrix.size(-1))
            # [N, S, num_heads, dk] * [N, S, num_heads, d_relation]
            query = self.relation_linear(query)
            # N * S parallel multiplications of [num_heads, d_relation] and [d_relation, S] matrices
            # [N, S, num_heads, d_relation] * [N, S, d_relation, S] -> [N, S, num_heads, S],
            # transpose to [N, num_heads, S, S]
            score2 = torch.matmul(query.transpose(1, 2), relation_bias_matrix.transpose(-2, -1)).transpose(1, 2)

        scores = (score1 + score2) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        if output is not None:
            output['attention'].append(p_attn)

        return self.output_linear(x)



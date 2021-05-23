import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, embed_size):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        removed dropout layer; dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        self.token = nn.Embedding(vocab_size, embedding_dim=embed_size)

    def forward(self, sequence, segment_label=None):
        x = self.token(sequence)
        return x
        # return self.dropout(x)


class RelationEmbedding(nn.Module):
    """
    """
    def __init__(self, embed_size, max_relative_1d_positions):
        """
        :param embed_size: embedding size of token embedding
        # :param dropout: dropout rate
        """
        super().__init__()

        self.vocab_1d_size = max_relative_1d_positions+1
        self.relative_1d_embed = nn.Embedding(self.vocab_1d_size, embedding_dim=embed_size)

        self.embed_size = embed_size
        self.max_relative_1d_positions = max_relative_1d_positions

    def forward(self, device, batch_size, seq_len, layer_attention_span):
        """
        """
        # get relative 1d position matrix
        range_vec = torch.arange(seq_len, device=device)
        relative_1d_dist_mat = torch.abs(range_vec[None, :] - range_vec[:, None])
        relative_1d_dist_mat_clipped = torch.clamp(relative_1d_dist_mat, min=0,
                                                   max=layer_attention_span)

        x = self.relative_1d_embed(relative_1d_dist_mat_clipped)

        # broadcast [S, S, dk] -> [N, S, S, dk]
        # x = x.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return x


class RelativeAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, num_heads, d_model, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, relation_bias_matrix, output=None):
        """
        forward function for RelativeAttention.
        :param query: [N, S, E]
        :param key:  [N, S, E]
        :param value: [N, S, E]
        :param mask: Optional
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
        # S parallel multiplications of [N*num_heads, dk] and [dk, S] matrices
        # [S, N*num_heads, dk] * [S, dk, S] -> [S, N*num_heads, S], transpose to [N, num_heads, S, S]
        query = query.view(batch_size*self.num_heads, -1, -1).transpose(0, 1)
        score2 = torch.matmul(query, relation_bias_matrix.transpose(-2, -1)).transpose(0, 1)
        score2 = score2.view(batch_size, self.num_heads, -1, -1)

        scores = (score1 + score2) / math.sqrt(self.d_k)

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


class TransformerEncoderLayer(nn.Module):
    """
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()

        self.attention = RelativeAttention(num_heads=attn_heads, d_model=hidden, dropout=dropout)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.linear1 = nn.Linear(hidden, feed_forward_hidden)
        self.linear2 = nn.Linear(feed_forward_hidden, hidden)
        # self.activation = F.gelu()
        self.activation = nn.ReLU()

    def forward(self, x, mask, relation_bias_matrix, output=None):
        x_att = self.attention(x, x, x, mask, relation_bias_matrix, output=output)

        # sublayer residue connection and the LayerNorm, follows the Pytorch transformer implementation
        x = self.norm1(x + self.dropout1(x_att))

        x_ff = self.linear2(self.dropout2(self.activation(self.linear1(x))))

        x = self.norm2(x + self.dropout3(x_ff))

        if output is not None:
            output['x_att'].append(x_att)
        return x


class ProEncoder(nn.Module):
    """
    Language Model for proteins
    """

    def __init__(self, vocab_size, hidden=512, n_layers=5, attn_heads=1, dropout=0.0,
                 max_relative_1d_positions=300,
                 visual=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.visual = visual

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # 2*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.relation_embedding = RelationEmbedding(hidden, max_relative_1d_positions)
        self.attention_span_list = [10, 30, 100, 300, 300]

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(hidden, attn_heads, self.feed_forward_hidden, dropout)
             for _ in range(n_layers)])

    def forward(self, x):
        """
        :param x: [N, S]  ->  (N, S, E)
        :return:
        """
        batch_size, seq_len = x.size()

        # attention masking for language model, keep the lower-half of the matrix
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, requires_grad=False))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # # attention masking for padded token
        # # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.token_embedding(x)  # [N, S, E]

        # running over multiple transformer blocks
        if self.visual:
            print('visualization')
            output = {'attention': [], 'x_att': []}
        else:
            output = None
        # print(output)

        i = 0
        for transformer in self.transformer_blocks:
            # relation_bias_matrix -- [N, S, S, E]
            relation_bias_matrix = self.relation_embedding(x.device, batch_size, seq_len, self.attention_span_list[i])
            x = transformer.forward(x, mask, relation_bias_matrix, output=output)

        if output is None:
            return x
        else:
            return x, output


class ProLM(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, vocab_size, hidden, n_layers, attn_heads=1, dropout=0.0):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.prolm = ProEncoder(vocab_size, hidden, n_layers, attn_heads, dropout=dropout)
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.prolm(x)
        x = self.softmax(self.linear(x))
        return x


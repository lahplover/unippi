import torch.nn as nn
import torch
import torch.nn.functional as F
import math


# copied the tensor2tensor code (Music transformer)
# def _relative_position_to_absolute_position_masked(x):
#   """Helper to dot_product_self_attention_relative_v2.
#   Rearrange an attention logits or weights Tensor.
#   The dimensions of the input represent:
#   [batch, heads, query_position, memory_position - query_position + length - 1]
#   The dimensions of the output represent:
#   [batch, heads, query_position, memory_position]
#   Only works with masked_attention.  Undefined behavior for regions of the
#   input where memory_position > query_position.
#   Args:
#     x: a Tensor with shape [batch, heads, length, length]
#   Returns:
#     a Tensor with shape [batch, heads, length, length]
#   """
#   batch, heads, length, _ = common_layers.shape_list(x)
#   x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
#   x = tf.reshape(x, [batch, heads, 1 + length, length])
#   x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
#   return x

# # copy and edit the Adaptive span code
# # B = batch_size, H = hidden_size, M = block_size, L = attn_span
#
# def _skew(X, pad_value):
#     """shift every row 1 step to right"""
#     # X = B x M x L
#     B, M, L = X.size()
#     X = F.pad(X, (0, M - L + 1), value=pad_value)  # B x M x (M+1)
#     X = X.view(B, -1)  # B x MM+M
#     X = X[:, :-M]  # B x MM
#     X = X.view(B, M, M)  # B x M x M
#     return X
#
# def _unskew(X, L):
#     """reverse _skew operation"""
#     # X = B x M x M
#     B, M, M = X.size()
#     X = X.view(B, -1)  # B x MM
#     X = F.pad(X, (0, M))  # B x (MM+M)
#     X = X.view(B, M, M + 1)  # B x M x (M+1)
#     X = X[:, :, :L]  # B x M x L
#     return X
#
# class SeqAttention(nn.Module):
#     """Sequential self-attention layer.
#     Each token will attend to its previous fixed number of steps.
#     Note that attention doesn't include the current step itself.
#     """
#     def __init__(self, hidden_size, attn_span,
#                  dropout):
#         nn.Module.__init__(self)
#         self.hidden_size = hidden_size # size of a single head
#         self.attn_span = attn_span
#
#     def forward(self, query, key, value, key_pe):
#         # query size = B x M x H
#         # key, value sizes = B x M x H
#
#         # compute attention from context
#         # B x M (dest) x (M+L) (src)
#         attn_cont = torch.matmul(query, key.transpose(-1, -2))
#         attn_cont = _unskew(attn_cont)  # B x M x L
#
#         # compute the effect of position embedding
#         attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
#         attn = attn_cont + attn_pos
#
#         attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
#         attn = F.softmax(attn, dim=-1)
#
#         attn_cont = _skew(attn, 0)  # B x M X (L+M)
#         out = torch.matmul(attn_cont, value)  # B x M x H
#
#         return out


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


def _unskew(x, L):
    # L is the relative attention span
    # (N, S, S) -> (N, S, L)
    N, S, S = x.size()
    x = x.view(N, -1)  # N x SS
    x = F.pad(x, (S, 0))  # N x (SS+S)
    x = x.view(N, S, S + 1)  # N x S x (S+1)
    x = x[:, :, -L:]  # N x S x L
    return x


def _skew(x, pad_value=0):
    # X = N x S x L  -> (N, S, S)
    N, S, L = x.size()
    x = F.pad(x, (S - L + 1, 0), value=pad_value)  # N x S x (S+1)
    x = x.view(N, -1)  # N x (SS+S)
    x = x[:, S:]  # N x SS
    x = x.view(N, S, S)  # N x S x S
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

    def forward(self, query, key, value, key_pe, attn_span, output=None):
        """
        forward function for RelativeAttention.
        :param query: [N, S, E]
        :param key:  [N, S, E]
        :param value: [N, S, E]
        :param attn_span: int; If attn_span < seq_len, attention is restricted to attn_span,
        i.e. attention outside attn_span is zero.
        If attn_span == seq_len, relative attention covers the whole sequence (positions
        beyond the max relative position are clipped to max relative position.)
        :return:
        """
        batch_size, seq_len, d_model = query.size()

        # attention masking for language model, keep the lower-half of the matrix
        mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device, requires_grad=False))
        mask = mask.unsqueeze(0)  # (1, S, S)

        # 1) Do all the linear projections in batch, reshape from d_model => num_heads x d_k
        # query, key, value have shapes [N, num_heads, S, dk]
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # first attention term, [N, num_heads, S, dk] * [N, num_heads, dk, S] -> [N, num_heads, S, S]
        attn_cont = torch.matmul(query, key.transpose(-2, -1))
        attn_cont = attn_cont.view(batch_size*self.num_heads, seq_len, seq_len)
        attn_cont = attn_cont.masked_fill(mask == 0, 0)

        # second attention term (relative attention)
        # [N, num_heads, S, dk] * [dk, L] -> [N, num_heads, S, L],
        attn_pos = torch.matmul(query, key_pe.transpose(0, 1))
        attn_pos = attn_pos.view(batch_size*self.num_heads, seq_len, -1)  # (N*h, S, L)

        # TODO: force sparse attention?
        if attn_span < seq_len:
            attn_cont = _unskew(attn_cont, attn_span)
            # attn_pos = attn_pos.masked_fill(attn_cont == 0, 0)  # attn_pos should have the same mask with attn_cont

            attn = (attn_cont + attn_pos) / math.sqrt(self.d_k)  # (N*h, S, L)
            attn = attn.masked_fill(attn_cont == 0, -1e9)  # before softmax, masked value should be -inf instead of 0.
            attn = F.softmax(attn, dim=-1)

            attn = _skew(attn, 0)  # (N*h, S, S)
        else:
            attn_pos = _skew(attn_pos, 0)  # (N*h, S, S)

            attn = (attn_cont + attn_pos) / math.sqrt(self.d_k)  # (N*h, S, S)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = F.softmax(attn, dim=-1)

        attn = attn.view(batch_size, self.num_heads, seq_len, seq_len)

        if self.dropout is not None:
            attn = self.dropout(attn)

        x = torch.matmul(attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        if output is not None:
            output['attention'].append(attn)

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

    def forward(self, x, key_pe, attn_span, output=None):
        x_att = self.attention(x, x, x, key_pe, attn_span, output=output)

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

    def __init__(self, vocab_size=23, hidden=10, n_layers=5, attn_heads=1, dropout=0.0,
                 max_rel_pos=300, attn_span_list=None,
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

        self.max_rel_pos = max_rel_pos
        self.relation_embedding = nn.Embedding(max_rel_pos+1, embedding_dim=hidden)

        if attn_span_list is not None:
            self.attn_span_list = attn_span_list
        else:
            self.attn_span_list = [10, 30, 100, 300, 1000]
        assert(len(self.attn_span_list) == n_layers)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(hidden, attn_heads, self.feed_forward_hidden, dropout)
             for _ in range(n_layers)])

    def get_pos_embed(self, device, attn_span):
        range_vec = torch.arange(attn_span, device=device) * (-1) + attn_span - 1
        range_vec_clipped = torch.clamp(range_vec, min=0, max=self.max_rel_pos)
        key_pe = self.relation_embedding(range_vec_clipped)
        return key_pe

    def forward(self, x):
        """
        :param x: [N, S]  ->  (N, S, E)
        :return:
        """
        # batch_size, seq_len = x.size()
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
            key_pe = self.get_pos_embed(x.device, self.attn_span_list[i])
            x = transformer.forward(x, key_pe, self.attn_span_list[i], output=output)

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


def test_pro_encoder():
    # a = ProEncoder(n_layers=1, attn_span_list=[5])
    a = ProEncoder(n_layers=1, attn_span_list=[3])
    x = torch.arange(10).reshape(2, 5) + 1
    out = a.forward(x)


if __name__ == '__main__':
    test_pro_encoder()







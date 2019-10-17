import torch.nn as nn
import torch


class RelationEmbedding(nn.Module):
    """
    """
    def __init__(self, embed_size, relative_1d, max_relative_1d_positions,
                 relative_3d, relative_3d_vocab_size,
                 dropout=0.1):
        """
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        assert (relative_1d | relative_3d)
        self.relative_1d = relative_1d
        self.relative_3d = relative_3d

        if relative_1d:
            self.vocab_1d_size = 2*max_relative_1d_positions+1
            self.relative_1d_embed = nn.Embedding(self.vocab_1d_size, embedding_dim=embed_size)
        if relative_3d:
            self.relative_3d_embed = nn.Embedding(relative_3d_vocab_size, embedding_dim=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.max_relative_1d_positions = max_relative_1d_positions
        # self.relative_3d_size = relative_3d_size
        # self.relative_3d_step = relative_3d_step
        # self.max_relative_3d = relative_3d_size * relative_3d_step

    def forward(self, device, batch_size, seq_len, distance_matrix):
        """
        :param distance_matrix: [N, S, S]
        :return:
        """
        # device = batch_size.device()
        # print('relative device:', device)
        if self.relative_1d:
            # get relative 1d position matrix
            range_vec = torch.arange(seq_len, device=device)
            relative_1d_dist_mat = range_vec[None, :] - range_vec[:, None]
            relative_1d_dist_mat_clipped = torch.clamp(relative_1d_dist_mat, min=-self.max_relative_1d_positions,
                                                       max=self.max_relative_1d_positions)
            # Shift values to be >= 0. Each integer still uniquely identifies a relative
            # position difference.
            relative_1d_dist_mat_clipped = relative_1d_dist_mat_clipped + self.max_relative_1d_positions
            x1 = self.relative_1d_embed(relative_1d_dist_mat_clipped)

        if self.relative_3d:
            # get relative 3d position matrix
            # distance_matrix_clipped = torch.clamp(distance_matrix, max=self.max_relative_3d)
            # relative_3d_matrix = distance_matrix_clipped // self.relative_3d_step
            x3 = self.relative_3d_embed(distance_matrix)

        # broadcast [S, S, dk] + [N, S, S, dk]
        if self.relative_1d & self.relative_3d:
            x = x1 + x3
        elif self.relative_1d:
            x = x1.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif self.relative_3d:
            x = x3
        else:
            raise ValueError('Both relative_1d and relative_3d are False.')
        return self.dropout(x)

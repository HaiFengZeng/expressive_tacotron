import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from hparams import create_hparams

hparams = create_hparams()


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=hparams.num_heads,
                 is_masked=False,
                 use_dropout=False,
                 _style='conv'):
        super(MultiHeadAttention, self).__init__()

        # if query_dim != key_dim:
        #     raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")
        self.use_dropout = use_dropout
        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(data=[key_dim], requires_grad=True, dtype=torch.float32)
        self._dropout_p = dropout_p
        self._is_masked = is_masked
        self.v = nn.Parameter(torch.randn([num_units]))
        self.use_batchnorm = False
        self.use_residual = False

        self.query_layer = nn.Linear(query_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(query_dim, num_units, 1)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(key_dim, num_units, 1)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(key_dim, num_units, 1)
        self.bn = nn.BatchNorm1d(num_units)

    def __split_last_dim(self, x, heads=None):
        if heads is None:
            heads = self._h
        # return shape [batch, length_x, num_heads, dim_x/num_heads]
        size = x.size()
        new_size = size[:-1] + (heads, int(size[-1] / heads))
        x = x.view(*new_size)
        return x

    def __split_head(self, q, k, v):
        # return [batch,num_heads, length_x, dim_x/num_heads]
        qs = self.__split_last_dim(q).permute(0, 2, 1, 3)
        ks = self.__split_last_dim(k).permute(0, 2, 1, 3)
        vs = self.__split_last_dim(v).permute(0, 2, 1, 3)
        # vs = v.unsqueeze(1).repeat(1, self._h, 1, 1)
        return qs, ks, vs

    def __combine_head(self, x):
        # [batch, length_x,num_heads, dim_x/num_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        size = x.size()
        new_size = size[:-2] + (size[2] * size[3],)
        return x.view(*new_size)

    def forward(self, query, keys):
        Q = self.query_layer(query.permute(0, 2, 1)).permute(0, 2, 1)  # [B,L,Dq]
        K = self.key_layer(keys.permute(0, 2, 1)).permute(0, 2, 1)  # [B,L,Dk]
        V = self.value_layer(keys.permute(0, 2, 1)).permute(0, 2, 1)  # [B,L,Dk]
        Q, K, V = self.__split_head(Q, K, V)
        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        # chunk_size = int(self._num_units / self._h)
        # Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=1)
        # K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=1)
        # V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=1)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(2, 3))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        if self.use_dropout:
            attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        attention = self.__combine_head(attention)
        # residual connection
        if self.use_residual:
            attention += query
        # apply batch normalization
        if self.use_dropout:
            attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention

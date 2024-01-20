import copy
import math
from logging import getLogger
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from base.abstract_traffic_state_model import AbstractTrafficStateModel
from utils import loss


def repeat_list(values, n):
    repeated = values * ceil(n / len(values))
    repeated.sort()
    return repeated[:n]


def calculate_random_walk_matrix(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    # W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D), W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))

    return sym_norm_Adj_matrix


def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  # 1 means reachable; 0 means unreachable


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x)


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''

    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))


class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.gcn(x)))


class SpatialAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout):
        super().__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.nb_head = nb_head
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, adj, mask=None):
        """
        :param mask:
        :param adj:G,N,N,D
        :return:
        """
        num_adj, num_node, _, d_model = adj.shape
        query = self.fc_q(adj).contiguous().view(num_adj, num_node, num_node, self.nb_head, -1).permute(0, 1, 3, 2, 4)
        key = self.fc_k(adj).contiguous().view(num_adj, num_node, num_node, self.nb_head, -1).permute(0, 1, 3, 2, 4)
        value = self.fc_v(adj).contiguous().view(num_adj, num_node, num_node, self.nb_head, -1).permute(0, 1, 3, 2, 4)

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()
        x = x.view(num_adj, num_node, -1, self.nb_head * self.d_k)
        return self.fc(x)


class GraphAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout):
        super().__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.nb_head = nb_head
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, adj, mask=None):
        """
        :param mask:
        :param adj:G,N,N,D
        :return:
        """
        adj = adj.permute(1, 2, 0, 3)
        num_node, _, num_adj, d_model = adj.shape
        query = self.fc_q(adj).contiguous().view(num_node, num_node, num_adj, self.nb_head, -1).permute(0, 1, 3, 2, 4)
        key = self.fc_k(adj).contiguous().view(num_node, num_node, num_adj, self.nb_head, -1).permute(0, 1, 3, 2, 4)
        value = self.fc_v(adj).contiguous().view(num_node, num_node, num_adj, self.nb_head, -1).permute(0, 1, 3, 2, 4)

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()
        x = x.view(num_node, num_node, -1, self.nb_head * self.d_k)
        return self.fc(x).permute(2, 0, 1, 3)


class GraphAttentionConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class GraphGatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_xs = nn.Linear(d_model, d_model)
        self.fc_xt = nn.Linear(d_model, d_model)
        self.fc_h = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, HS, HG):
        """
        :param HS: G,N,N,D
        :param HG: G,N,N,D
        :return: G,N,N,D
        """
        xs = self.fc_xs(HS)
        hs = self.fc_xt(HG)
        z = torch.sigmoid(torch.add(xs, hs))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HG))
        H = self.fc_h(H)
        return H


class GraphAttBlock(nn.Module):
    def __init__(self, n_head, d_model, graph_num, node_num, dropout=.0):
        super().__init__()
        self.fc_1 = nn.Linear(1, d_model)
        self.spatialAttention = SpatialAttention(n_head, d_model, dropout)
        self.graphAttention = GraphAttention(n_head, d_model, dropout)
        self.graphGatedFusion = GraphGatedFusion(d_model)
        self.fc_2 = nn.Linear(d_model, 1)
        self.adj_w = nn.Parameter(torch.randn(graph_num, node_num, node_num))
        self.relu = nn.ReLU()

    def forward(self, *adj):
        adj = torch.stack(adj, dim=0).unsqueeze(-1).to(adj[0].device)
        adj = self.fc_1(adj)
        hs = self.spatialAttention(adj)
        hg = self.graphAttention(adj)
        h = self.graphGatedFusion(hs, hg)
        h = self.fc_2(h).squeeze(-1)
        adj = self.relu(torch.sum(self.adj_w * h, dim=0))
        return adj


class SpatialAttentionScaledGCN(nn.Module):
    def __init__(self, n_head, d_model, K, in_channels, out_channels, dropout=.0, adj=None):
        super(SpatialAttentionScaledGCN, self).__init__()
        self.adj = adj  # (N, N)
        self.graph_num = len(adj)
        self.num_node = self.adj[0].shape[0]
        self.SG_ATT = GraphAttBlock(n_head, d_model, self.graph_num, self.num_node, dropout=dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta1 = nn.Linear(out_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        adj = self.SG_ATT(*self.adj)
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta1(self.Theta(torch.matmul(adj.mul(spatial_attention), x))).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class SpatialAttentionGCN(nn.Module):
    def __init__(self, n_head, d_model, K, in_channels, out_channels, dropout=.0, adj=None):
        super().__init__()
        self.adj = adj
        self.graph_num = len(adj)
        self.num_node = self.adj[0].shape[0]
        self.SG_ATT = GraphAttBlock(n_head, d_model, self.graph_num, self.num_node, dropout=dropout)
        self.K = K
        self.relu = nn.ReLU()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])

    def forward(self, x):
        """
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        """
        adj = self.SG_ATT(*self.adj)
        cheb_polynomials = self._cheb_polynomial(adj, self.K)
        batch_size, N, T, F_in = x.shape
        outputs = []

        for time_step in range(T):

            graph_signal = x[:, :, time_step, :]  # (b, N, F_in)

            output = torch.zeros(batch_size, N, self.out_channels).to(x.device)  # (b, N, F_out)

            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1)).permute(0, 1, 3, 2)

    @staticmethod
    def _cheb_polynomial(L_tilde, K):
        N = L_tilde.shape[0]

        cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]

        for i in range(2, K):
            cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

        return cheb_polynomials


class MultiCoreAttentionAwareTemporalContex_qc_kc(nn.Module):
    def __init__(self, nb_head, d_model, kernel_sizes=None, dropout=.0):
        super().__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.nb_head = nb_head
        self.kernel_sizes = repeat_list(kernel_sizes, self.nb_head)
        self.padding = [i - 1 for i in self.kernel_sizes]
        self.fc_q = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding)])
        self.fc_k = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding)])
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_h = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        queries = torch.split(query, self.d_k, dim=-1)
        keys = torch.split(key, self.d_k, dim=-1)
        query = torch.stack(
            [self.fc_q[i](queries[i].permute(0, 3, 1, 2))[:, :, :, :-self.padding[i]].contiguous() for i in
             range(self.nb_head)], dim=-1).permute(0, 2, 4, 3, 1)
        key = torch.stack(
            [self.fc_k[i](keys[i].permute(0, 3, 1, 2))[:, :, :, :-self.padding[i]].contiguous() for i in
             range(self.nb_head)], dim=-1).permute(0, 2, 4, 3, 1)

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.fc_v(value).view(nbatches, N, -1, self.nb_head, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.nb_head * self.d_k)  # (batch, N, T1, d_model)
        return self.fc_h(x)


class MultiCoreAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, kernel_sizes=None, dropout=.0):
        super().__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.nb_head = nb_head
        self.kernel_sizes = repeat_list(kernel_sizes, self.nb_head)
        self.padding = [(i - 1) // 2 for i in self.kernel_sizes]
        self.fc_q = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding)])
        self.fc_k = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding)])
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_h = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        queries = torch.split(query, self.d_k, dim=-1)
        keys = torch.split(key, self.d_k, dim=-1)
        query = torch.stack(
            [self.fc_q[i](queries[i].permute(0, 3, 1, 2)).contiguous() for i in range(self.nb_head)],
            dim=-1).permute(0, 2, 4, 3, 1)
        key = torch.stack(
            [self.fc_k[i](keys[i].permute(0, 3, 1, 2)).contiguous() for i in range(self.nb_head)],
            dim=-1).permute(0, 2, 4, 3, 1)

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.fc_v(value).view(nbatches, N, -1, self.nb_head, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.nb_head * self.d_k)  # (batch, N, T1, d_model)
        return self.fc_h(x)


class MultiCoreAttentionAwareTemporalContex_qc_k1d(nn.Module):
    def __init__(self, nb_head, d_model, kernel_sizes=None, dropout=.0):
        super().__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.nb_head = nb_head
        self.kernel_sizes = repeat_list(kernel_sizes, self.nb_head)
        self.padding_k = [(i - 1) // 2 for i in self.kernel_sizes]
        self.padding = [i - 1 for i in self.kernel_sizes]
        self.fc_q = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding)])
        self.fc_k = nn.ModuleList(
            [nn.Conv2d(self.d_k, self.d_k, (1, kernel_size), padding=(0, padding)) for kernel_size, padding in
             zip(self.kernel_sizes, self.padding_k)])
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_h = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        queries = torch.split(query, self.d_k, dim=-1)
        keys = torch.split(key, self.d_k, dim=-1)
        query = torch.stack(
            [self.fc_q[i](queries[i].permute(0, 3, 1, 2))[:, :, :, :-self.padding[i]].contiguous() for i in
             range(self.nb_head)], dim=-1).permute(0, 2, 4, 3, 1)
        key = torch.stack(
            [self.fc_k[i](keys[i].permute(0, 3, 1, 2)).contiguous() for i in range(self.nb_head)],
            dim=-1).permute(0, 2, 4, 3, 1)

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.fc_v(value).view(nbatches, N, -1, self.nb_head, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.nb_head * self.d_k)  # (batch, N, T1, d_model)
        return self.fc_h(x)


class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):  # key causal; query causal;
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        '''
        :param nb_head:
        :param d_model:
        :param kernel_size:
        :param dropout:
        '''
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query, key = [
            l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                -1).permute(0, 3, 1, 4, 2) for l, x in
            zip(self.conv1Ds_aware_temporal_context, (query, key))]

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1) // 2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query, key = [
            l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            for l, x in zip(self.conv1Ds_aware_temporal_context,
                            (query, key))]

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):  # query: causal conv; key 1d conv
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                              padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                            padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N,
        # T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[
                :, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(
            0, 3, 1, 4, 2)
        key = self.key_conv1Ds_aware_temporal_context(
            key.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N,
                                                       -1).permute(0, 3, 1, 4, 2)

        # deal with value: (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h,
        # d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False,
                                                             key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False,
                                                            key_multi_segment=True))  # output: (batch, N, T', d_model)
            return self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False,
                               key_multi_segment=False)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False,
                              key_multi_segment=True)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


class STGCL(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.dropout = config.get("dropout", 0)
        self.kernel_size = config.get("kernel_size", 3)
        self.kernel_sizes = config.get("kernel_sizes", [3, 5, 7])
        self.multiCore = config.get("multiCore", True)
        self.scale = config.get("scale", True)
        self.layer_num = config.get("num_layers", 1)
        # self.num_layers = config.get("num_layers", 1)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.d_model = config.get("d_model", 64)
        self.nb_head = config.get('nb_head', 8)
        self.max_len = config.get("max_len", 5000)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.input_dim = self.data_feature.get("feature_dim", 1)
        self.K = config.get("K", 3)
        self.k = config.get("k", 2)
        self.adj_mx = self.data_feature.get("adj_mx")
        self.device = config.get('device', torch.device('cpu'))
        self.adj_dtw = torch.FloatTensor(self.data_feature.get("adj_dtw")).to(self.device)
        self.adj_SSG = torch.FloatTensor(calculate_random_walk_matrix(self.adj_mx)).to(self.device)
        self.adj = torch.FloatTensor(
            calculate_random_walk_matrix(self._calculate_k_order_neighbor_matrix(self.adj_mx, self.k))).to(self.device)
        self.adj_SSG = self.adj_SSG * self.adj_dtw * self.adj_SSG

        src_dense = nn.Linear(self.input_dim, self.d_model)
        trg_dense = nn.Linear(self.output_dim, self.d_model)
        encode_temporal_position = TemporalPositionalEncoding(self.d_model, self.dropout, self.max_len)
        decode_temporal_position = TemporalPositionalEncoding(self.d_model, self.dropout, self.max_len)
        spatial_position = SpatialPositionalEncoding(self.d_model, self.adj.shape[0], self.dropout, )
        self.encoder_embedding = nn.Sequential(src_dense, copy.deepcopy(encode_temporal_position),
                                               copy.deepcopy(spatial_position))
        self.decoder_embedding = nn.Sequential(trg_dense, copy.deepcopy(decode_temporal_position),
                                               copy.deepcopy(spatial_position))
        if not self.multiCore:
            attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(self.nb_head, self.d_model, self.kernel_size,
                                                                    dropout=self.dropout)
            attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(self.nb_head, self.d_model, self.kernel_size,
                                                                   dropout=self.dropout)
            att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(self.nb_head, self.d_model, self.kernel_size,
                                                                 dropout=self.dropout)
        else:
            attn_ss = MultiCoreAttentionAwareTemporalContex_q1d_k1d(self.nb_head, self.d_model, self.kernel_sizes,
                                                                    dropout=self.dropout)
            attn_st = MultiCoreAttentionAwareTemporalContex_qc_k1d(self.nb_head, self.d_model, self.kernel_sizes,
                                                                   dropout=self.dropout)
            att_tt = MultiCoreAttentionAwareTemporalContex_qc_kc(self.nb_head, self.d_model, self.kernel_sizes,
                                                                 dropout=self.dropout)
        if self.scale:
            position_wise_gcn = PositionWiseGCNFeedForward(
                SpatialAttentionScaledGCN(self.nb_head, self.d_model, self.K, self.d_model, self.d_model, self.dropout,
                                          adj=(self.adj, self.adj_dtw, self.adj_SSG)),
                dropout=self.dropout)
        else:
            position_wise_gcn = PositionWiseGCNFeedForward(
                SpatialAttentionGCN(self.nb_head, self.d_model, self.K, self.d_model, self.d_model, self.dropout,
                                    adj=(self.adj, self.adj_dtw, self.adj_SSG)),
                dropout=self.dropout)

        encoderLayer = EncoderLayer(self.d_model, attn_ss, copy.deepcopy(position_wise_gcn), self.dropout,
                                    residual_connection=True, use_LayerNorm=True)

        self.encoder = Encoder(encoderLayer, self.layer_num)

        decoderLayer = DecoderLayer(self.d_model, att_tt, attn_st, copy.deepcopy(position_wise_gcn), self.dropout,
                                    residual_connection=True, use_LayerNorm=True)

        self.decoder = Decoder(decoderLayer, self.layer_num)

        self.generator = nn.Linear(self.d_model, self.output_dim)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        x = batch["X"].permute(0, 2, 1, 3)
        B, N, T, C = x.shape
        decoder_start_inputs = x[:, :, -1:, :self.output_dim]
        decoder_start_zeros = torch.zeros(B, N, self.output_window - 1, self.output_dim, device=x.device)
        decoder_input_list = [decoder_start_inputs, decoder_start_zeros]
        decoder_inputs = torch.cat(decoder_input_list, dim=2)
        encoder_output = self.encode(x)

        predict_output = self.decode(decoder_inputs, encoder_output)
        return predict_output.permute(0, 2, 1, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        h = self.encoder_embedding(src)
        return self.encoder(h)

    def decode(self, trg, encoder_output):
        return self.generator(self.decoder(self.decoder_embedding(trg), encoder_output))

    @classmethod
    def _calculate_k_order_neighbor_matrix(cls, adj, k):
        adj = adj + np.eye(adj.shape[0])
        k_adj = adj
        for i in range(1, k):
            k_adj = np.matmul(k_adj, adj)
        return k_adj

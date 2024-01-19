from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import linalg
from torch import nn, Tensor

from base.abstract_traffic_state_model import AbstractTrafficStateModel
from utils import loss
from torch.nn import functional as F


def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(d, d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)


class GraphConv_(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv_, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):

        b, n, input_dim = inputs.shape
        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = x0.unsqueeze(dim=0)  # (1, num_nodes, input_dim * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = support.mm(x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * support.mm(x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)


class EvolutionCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int, layer: int,
                 n_dim: int):
        super(EvolutionCell, self).__init__()
        self.layer = layer
        self.perceptron = nn.ModuleList()
        self.graphconv = nn.ModuleList()
        self.attlinear = nn.Linear(num_nodes * output_dim, 1)
        self.graphconv.append(GraphConv_(input_dim, output_dim, num_nodes, n_supports, max_step))
        for i in range(1, layer):
            self.graphconv.append(GraphConv_(output_dim, output_dim, num_nodes, n_supports, max_step))

    def forward(self, inputs, supports: List):
        outputs = []
        for i in range(self.layer):
            inputs = self.graphconv[i](inputs, [supports[i]])
            outputs.append(inputs)
        out = self.attention(torch.stack(outputs, dim=1))
        # out = outputs[-1]
        return out

    def attention(self, inputs: Tensor):
        b, g, n, f = inputs.size()
        x = inputs.reshape(b, g, -1)
        out = self.attlinear(x)  # (batch, graph, 1)
        weight = F.softmax(out, dim=1)

        outputs = (x * weight).sum(dim=1).reshape(b, n, f)
        return outputs


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int, e_layer: int,
                 n_dim: int):
        super(DCGRUCell, self).__init__()
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = EvolutionCell(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop,
                                            e_layer, n_dim)
        self.candidate_g_conv = EvolutionCell(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop,
                                              e_layer, n_dim)

    def forward(self, inputs: Tensor, supports: List[Tensor], states) -> Tuple[Tensor, Tensor]:
        """
        :param inputs: Tensor[Batch, Node, Feature]
        :param supports:
        :param states:Tensor[Batch, Node, Hidden_size]
        :return:
        """
        r_u = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c
        return outputs, new_state


class DCRNNEncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int, n_layers: int,
                 e_layer: int, n_dim: int):
        super(DCRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.append(DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop, e_layer, n_dim))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop, e_layer, n_dim))

    def forward(self, inputs: Tensor, supports: List[Tensor]) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, input_size]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [n_layers, B, N, hidden_size]
        """

        b, t, n, _ = inputs.shape
        dv, dt = inputs.device, inputs.dtype

        states = list(torch.zeros(len(self), b, n, self.hidden_size, device=dv, dtype=dt))
        inputs = list(inputs.transpose(0, 1))

        for i_layer, cell in enumerate(self):
            for i_t in range(t):
                inputs[i_t], states[i_layer] = cell(inputs[i_t], supports, states[i_layer])

        return torch.stack(states)


class DCRNNDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, num_node: int,
                 n_supports: int, k_hop: int, n_layers: int, n_preds: int, e_layer: int, n_dim: int):
        super(DCRNNDecoder, self).__init__()
        self.output_size = output_size
        self.n_preds = n_preds
        self.append(DCGRUCell(output_size, hidden_size, num_node, n_supports, k_hop, e_layer, n_dim))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop, e_layer, n_dim))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, supports: List[Tensor], states: Tensor) -> Tensor:
        """
        :param supports: list of sparse tensors, each of shape [N, N]
        :param states: tensor, [n_layers, B, N, hidden_size]
        :param targets: None or tensor, [B, T, N, output_size]
        :param teacher_force: random to use targets as decoder inputs
        :return: tensor, [B, T, N, output_size]
        """
        n_layers, b, n, _ = states.shape

        inputs = torch.zeros(b, n, self.output_size, device=states.device, dtype=states.dtype)

        states = list(states)
        assert len(states) == n_layers

        new_outputs = list()
        for i_t in range(self.n_preds):
            for i_layer in range(n_layers):
                inputs, states[i_layer] = self[i_layer](inputs, supports, states[i_layer])
            inputs = self.out(inputs)
            new_outputs.append(inputs)

        return torch.stack(new_outputs, 1)


class EvoNN2(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = normalized_laplacian(self.data_feature.get('adj_dtw'))
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.hidden_size = config.get("hidden_size")
        self.n_dim = config.get("n_dim")
        self.n_supports = config.get("n_supports")
        self.k_hop = config.get("k_hop")
        self.n_rnn_layers = config.get("n_rnn_layers")
        self.n_gconv_layers = config.get("n_gconv_layers")
        self.device = config.get('device', torch.device('cpu'))
        self.adj_mx = torch.tensor(self.adj_mx, device=self.device, dtype=torch.float)

        m, p, n = torch.svd(self.adj_mx)
        initemb1 = torch.mm(m[:, :self.n_dim], torch.diag(p[:self.n_dim] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:self.n_dim] ** 0.5), n[:, :self.n_dim].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)

        self.encoder = DCRNNEncoder(self.feature_dim, self.hidden_size, self.num_nodes, self.n_supports, self.k_hop,
                                    self.n_rnn_layers, self.n_gconv_layers,
                                    self.n_dim)
        self.decoder = DCRNNDecoder(self.output_dim, self.hidden_size, self.num_nodes, self.n_supports, self.k_hop,
                                    self.n_rnn_layers, self.output_window,
                                    self.n_gconv_layers, self.n_dim)

        self.w1 = nn.Parameter(torch.eye(self.n_dim), requires_grad=True)
        self.w2 = nn.Parameter(torch.eye(self.n_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(self.n_dim), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(self.n_dim), requires_grad=True)
        self.graph0 = None
        self.graph1 = None
        self.graph2 = None

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

    def forward(self, batch):
        inputs = batch["X"]
        graph = list()
        nodevec1 = self.nodevec1
        nodevec2 = self.nodevec2
        n = nodevec1.size(0)
        self.graph0 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        graph.append(self.graph0)

        nodevec1 = nodevec1.mm(self.w1) + self.b1.repeat(n, 1)
        nodevec2 = (nodevec2.T.mm(self.w1) + self.b1.repeat(n, 1)).T
        self.graph1 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        graph.append(self.graph1)
        nodevec1 = nodevec1.mm(self.w2) + self.b2.repeat(n, 1)
        nodevec2 = (nodevec2.T.mm(self.w2) + self.b2.repeat(n, 1)).T
        self.graph2 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        graph.append(self.graph2)
        states = self.encoder(inputs, graph)
        outputs = self.decoder(graph, states)
        return outputs

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

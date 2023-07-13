import warnings
from logging import getLogger

import torch
import torchcde
import torchdiffeq
from torch import nn
from torch.nn import functional as F

from base.abstract_traffic_state_model import AbstractTrafficStateModel
from utils import loss


class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                     for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)  # 32,32*4  -> # 32,32,4

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class VectorField_g(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k,
                 embed_dim, g_type):
        super(VectorField_g, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        # FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4
        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                          hidden_channels * hidden_channels)  # 32,32*4  -> # 32,32,4

        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        # FIXME:
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z  # torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # laplacian=False
        laplacian = False
        if laplacian:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return z


class STG_NCDE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.input_dim = self.data_feature.get("feature_dim", 1) + 1
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.num_nodes = self.data_feature.get('num_nodes', 200)
        self.device = config.get('device', torch.device('cpu'))
        self.time = torch.linspace(0, self.input_window - 1, self.input_window).to(self.device)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.hidden_hidden_dim = config.get("hidden_hidden_dim", 64)
        self.num_layers = config.get("num_layers", 3)
        self.cheb_k = config.get("cheb_k", 2)
        self.embed_dim = config.get("embed_dim", 10)
        self.g_type = config.get("g_type", "agc")
        self.func_f = FinalTanh_f(self.input_dim, self.hidden_dim, self.hidden_hidden_dim, self.num_layers)
        self.func_g = VectorField_g(self.input_dim, self.hidden_dim, self.hidden_hidden_dim, self.num_layers,
                                    self.num_nodes, self.cheb_k, self.embed_dim, self.g_type)
        self.solver = config.get("solver", "rk4")
        self.atol = config.get("atol", 1e-9)
        self.rtol = config.get("rtol", 1e-7)
        self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self.pred_real_value = config.get("pred_real_value", False)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        coeffs = batch["X"]
        X = torchcde.CubicSpline(coeffs)
        h0 = self.initial_h(X.evaluate(X.interval[0]))
        z0 = self.initial_z(X.evaluate(X.interval[0]))
        z_t = cdeint(X=X, h0=h0, z0=z0, func_f=self.func_f, func_g=self.func_g, t=self.time, method=self.solver,
                     atol=self.atol, rtol=self.rtol)
        z_T = z_t[-1:, ...].transpose(0, 1)

        # CNN based predictor
        output = self.end_conv(z_T)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        if self.pred_real_value:
            y_predicted = y_predicted[..., :self.output_dim]
        else:
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)


class _VectorField(nn.Module):
    def __init__(self, X, func_f, func_g):
        super(_VectorField, self).__init__()

        self.X = X
        self.func_f = func_f
        self.func_g = func_g

        # torchsde backend
        # self.sde_type = getattr(func, "sde_type", "stratonovich")
        # self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, hz):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)

        # vector_field is of shape (..., hidden_channels, input_channels)
        h = hz[0]  # h: torch.Size([64, 207, 32])
        z = hz[1]  # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h)  # vector_field_f: torch.Size([64, 207, 32, 2])
        vector_field_g = self.func_g(z)  # vector_field_g: torch.Size([64, 207, 32, 2])
        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        return tuple([dh, out])


def cdeint(X, func_f, func_g, h0, z0, t, adjoint=True, backend="torchdiffeq", **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcde.CubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be a callable describing the vector field f(t, z). If using `adjoint=True` (the default), then
            should be an instance of `torch.nn.Module`, to collect the parameters for the adjoint pass. Will be called
            with a scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions. If it
            has a method `prod` then that will be called to calculate the matrix-vector product f(t, z) dX_t/dt, via
            `func.prod(t, z, dXdt)`.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        backend: Either "torchdiffeq" or "torchsde". Which library to use for the solvers. Note that if using torchsde
            that the Brownian motion component is completely ignored -- so it's still reducing the CDE to an ODE --
            but it makes it possible to e.g. use an SDE solver there as the ODE/CDE solver here, if for some reason
            that's desired.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq (the most common are `rtol`, `atol`,
            `method`, `options`) or the sdeint solver of torchsde.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Note:
        Supports tupled input, i.e. z0 can be a tuple of tensors, and X.derivative and func can return tuples of tensors
        of the same length.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.odeint` or `torchsde.sdeint`.
    """

    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if adjoint:
        if "adjoint_atol" not in kwargs:
            kwargs["adjoint_atol"] = kwargs["atol"]
        if "adjoint_rtol" not in kwargs:
            kwargs["adjoint_rtol"] = kwargs["rtol"]

    if adjoint and 'adjoint_params' not in kwargs:
        for buffer in X.buffers():
            # Compare based on id to avoid PyTorch not playing well with using `in` on tensors.
            if buffer.requires_grad:
                warnings.warn("One of the inputs to the control path X requires gradients but "
                              "`kwargs['adjoint_params']` has not been passed. This is probably a mistake: these "
                              "inputs will not receive a gradient when using the adjoint method. Either have the input "
                              "not require gradients (if that was unintended), or include it (and every other "
                              "parameter needing gradients) in `adjoint_params`. For example:\n"
                              "```\n"
                              "coeffs = ...\n"
                              "func = ...\n"
                              "X = CubicSpline(coeffs)\n"
                              "adjoint_params = tuple(func.parameters()) + (coeffs,)\n"
                              "cdeint(X=X, func=func, ..., adjoint_params=adjoint_params)\n"
                              "```")

    vector_field = _VectorField(X=X, func_f=func_f, func_g=func_g)
    if backend == "torchdiffeq":
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        init0 = (h0, z0)
        out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    # batch_dims = range(1, len(out.shape) - 1)
    # out = out.permute(*batch_dims, 0, -1)
    return out[-1]

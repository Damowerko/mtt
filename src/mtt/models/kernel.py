import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torchcps.kernel.nn import GaussianKernel, KernelConv, KernelSample, Mixture

from mtt.models.sparse import SparseBase, SparseOutput
from mtt.utils import add_model_specific_args


class KernelEncoderLayer(nn.Module):
    def __init__(
        self,
        pos_dim: int,
        max_filter_kernels: int,
        n_channels: int,
        n_channels_mlp: int,
        sigma: float,
        update_positions: bool,
    ):
        super().__init__()
        self.conv = KernelConv(
            max_filter_kernels,
            n_channels,
            n_channels,
            pos_dim,
            kernel_spread=sigma,
            update_positions=update_positions,
            kernel_init="grid",
        )
        self.sample = KernelSample(
            kernel=GaussianKernel(sigma), nonlinearity=nn.LeakyReLU(), normalize=False
        )
        self.conv_norm = nn.BatchNorm1d(n_channels)
        self.mlp = MLP(
            [n_channels, n_channels_mlp, n_channels + pos_dim],
            act=nn.LeakyReLU(),
            norm="batch_norm",
            act_first=True,
        )
        self.nonlinearity = nn.LeakyReLU()

    def forward(self, x_in: Mixture):
        x = self.conv.forward(x_in)
        # KernelSample applies the nonlinearity after sampling
        x = self.sample.forward(x, x_in.positions, x_in.batch)
        x = x.map_weights(self.conv_norm)
        mlp_out = self.mlp.forward(x.weights)
        delta_positions, delta_weights = torch.split(
            mlp_out, [x.positions.shape[-1], x.weights.shape[-1]], dim=-1
        )
        x_out = Mixture(
            x.positions + delta_positions,
            x.weights + delta_weights,
            x.batch,
        )
        return x_out


class KernelDecoderLayer(nn.Module):
    def __init__(
        self,
        pos_dim: int,
        max_filter_kernels: int,
        n_channels: int,
        n_channels_mlp: int,
        sigma_cross: float,
        sigma_self: float,
        update_positions: bool,
    ):
        super().__init__()
        self.cross_conv = KernelConv(
            max_filter_kernels,
            n_channels,
            n_channels,
            pos_dim,
            kernel_spread=sigma_cross,
            update_positions=update_positions,
            kernel_init="grid",
        )
        self.sample_cross = KernelSample(
            kernel=GaussianKernel(sigma_cross),
            nonlinearity=nn.LeakyReLU(),
            normalize=False,
        )
        self.norm_cross = nn.BatchNorm1d(n_channels)
        self.conv_self = KernelConv(
            max_filter_kernels,
            n_channels,
            n_channels,
            pos_dim,
            kernel_spread=sigma_self,
            update_positions=update_positions,
            kernel_init="grid",
        )
        self.sample_self = KernelSample(
            kernel=GaussianKernel(sigma_self),
            nonlinearity=nn.LeakyReLU(),
            normalize=False,
        )
        self.norm_self = nn.BatchNorm1d(n_channels)
        self.mlp_self = MLP(
            [n_channels, n_channels_mlp, n_channels + pos_dim],
            act=nn.LeakyReLU(),
            norm="batch_norm",
            act_first=True,
            plain_last=True,
        )

    def forward(self, z: Mixture, e: Mixture):
        # convolve e and sample at z.positions
        e_conv = self.cross_conv.forward(e)
        e_sampled = self.sample_cross.forward(e_conv, z.positions, z.batch)
        e_sampled = e_sampled.map_weights(self.norm_cross)

        # z = z + e_sampled
        z = z.map_weights(e_sampled.weights.add)
        # z -> conv -> sample -> norm -> +z
        z_conv = self.conv_self.forward(z)
        z_sampled = self.sample_self.forward(z_conv, z.positions, z.batch)
        z_sampled = z_sampled.map_weights(self.norm_self)
        z = z.map_weights(z_sampled.weights.add)
        # use mlp to update z weights and positions
        mlp_out = self.mlp_self.forward(z.weights)
        delta_positions, delta_weights = torch.split(
            mlp_out, [z.positions.shape[-1], z.weights.shape[-1]], dim=-1
        )
        z = Mixture(
            z.positions + delta_positions,
            z.weights + delta_weights,
            z.batch,
        )
        return z


class KNNBlock(nn.Module):
    def __init__(
        self,
        conv: KernelConv,
        sample: KernelSample,
        norm: nn.Module,
        delta_module: nn.Module,
    ):
        super().__init__()
        self.conv = conv
        self.sample = sample
        self.norm = norm
        self.delta_module = delta_module

    def forward(
        self,
        x: Mixture,
    ):
        # convolution and sampling operation
        y = self.conv.forward(x)
        y = y.map_weights(self.norm)
        y = self.sample.forward(y, x.positions, x.batch)

        # pointwise update of the kernel weights and positions
        delta_out = self.delta_module.forward(x.weights)
        delta_positions, delta_weights = torch.split(
            delta_out, [x.positions.shape[-1], x.weights.shape[-1]], dim=-1
        )
        y = Mixture(
            x.positions + delta_positions,
            x.weights + delta_weights,
            x.batch,
        )
        return y


class KNNModule(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        n_layers_mlp: int,
        hidden_channels_mlp: int,
        sigma: float | typing.Sequence[float],
        max_filter_kernels: int,
        update_positions: bool,
        alpha: float | None = None,
        kernel_init: str = "uniform",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.nonlinearity = nn.LeakyReLU()

        self.readin = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels_mlp,
            out_channels=hidden_channels,
            num_layers=n_layers_mlp,
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=False,
        )
        self.readout = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels_mlp,
            out_channels=out_channels,
            num_layers=n_layers_mlp,
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=True,
        )
        sigma = sigma if isinstance(sigma, typing.Sequence) else [sigma] * n_layers

        blocks: list[KNNBlock] = []
        for l in range(self.n_layers):
            blocks += [
                KNNBlock(
                    conv=KernelConv(
                        max_filter_kernels=max_filter_kernels,
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_dimensions=n_dimensions,
                        kernel_spread=sigma[l],
                        update_positions=update_positions,
                        kernel_init=kernel_init,
                    ),
                    sample=KernelSample(
                        kernel=GaussianKernel(sigma[l]),
                        alpha=alpha,
                        nonlinearity=nn.LeakyReLU(),
                    ),
                    norm=nn.BatchNorm1d(hidden_channels),
                    delta_module=MLP(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels_mlp,
                        out_channels=hidden_channels + n_dimensions,
                        num_layers=n_layers_mlp,
                        act=self.nonlinearity,
                        norm="batch_norm",
                        plain_last=True,
                    ),
                )
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Mixture):
        x = x.map_weights(self.readin.forward)
        for l in range(self.n_layers):
            x = typing.cast(KNNBlock, self.blocks[l]).forward(x)
        x = x.map_weights(self.readout.forward)
        return x


class KNN(SparseBase):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        measurement_dim: int,
        state_dim: int,
        pos_dim: int,
        hidden_channels: int = 32,
        n_layers: int = 4,
        n_layers_mlp: int = 2,
        hidden_channels_mlp: int = 128,
        sigma: float = 1.0,
        max_filter_kernels: int = 32,
        update_positions: bool = True,
        alpha: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = KNNModule(
            pos_dim,
            in_channels=measurement_dim * self.input_length,
            hidden_channels=hidden_channels,
            out_channels=state_dim + 1,
            n_layers=n_layers,
            n_layers_mlp=n_layers_mlp,
            hidden_channels_mlp=hidden_channels_mlp,
            sigma=sigma,
            max_filter_kernels=max_filter_kernels,
            update_positions=update_positions,
            alpha=alpha,
        )

    def _forward(self, x: torch.Tensor, x_pos: torch.Tensor, x_batch: torch.Tensor):
        x_mixture = Mixture(x_pos, x, x_batch)
        y_mixture = self.model.forward(x_mixture)
        mu = y_mixture.positions
        sigma = y_mixture.weights[:, :-1]
        logits = y_mixture.weights[:, -1]
        batch = y_mixture.batch
        return mu, sigma, logits, batch

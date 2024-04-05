import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.pool.select import SelectOutput, SelectTopK
from torchcps.kernel.nn import (
    GaussianKernel,
    KernelConv,
    KernelSample,
    Mixture,
    solve_kernel,
)

from mtt.data.sparse import SparseData
from mtt.models.sparse import SparseBase, SparseInput, SparseOutput
from mtt.utils import add_model_specific_args, to_polar_torch


class KernelEncoderLayer(nn.Module):
    def __init__(
        self,
        max_filter_kernels: int,
        n_channels: int,
        n_channels_mlp: int,
        sigma: float,
        update_positions: bool,
        select_ratio: float = 1.0,
    ):
        super().__init__()
        pos_dim = 2
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
            kernel=GaussianKernel(sigma), nonlinearity=nn.LeakyReLU()
        )
        self.conv_norm = nn.BatchNorm1d(n_channels)
        self.mlp = gnn.MLP(
            [n_channels, n_channels_mlp, n_channels + pos_dim],
            act=nn.LeakyReLU(),
            norm="batch_norm",
            act_first=True,
            plain_last=True,
        )
        self.nonlinearity = nn.LeakyReLU()
        self.select = (
            SelectTopK(n_channels, select_ratio) if select_ratio < 0.99 else None
        )

    def forward(self, x_in: Mixture):
        x = self.conv.forward(x_in)
        # KernelSample applies the nonlinearity after sampling
        x = self.sample.forward(x, x_in.positions, x_in.batch)
        x = x.map_weights(self.conv_norm)
        # residual connection
        x = x.map_weights(x_in.weights.add)
        mlp_out = self.mlp.forward(x.weights)
        delta_positions, delta_weights = torch.split(
            mlp_out, [x.positions.shape[-1], x.weights.shape[-1]], dim=-1
        )
        # residual connection
        x_out = Mixture(
            x.positions + delta_positions,
            x.weights + delta_weights,
            x.batch,
        )
        if self.select is not None:
            # see: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.TopKPooling.html
            # we have to multiply the output by the score to make this differentiable
            assert x_out.batch is not None
            batch_idx = torch.repeat_interleave(
                torch.arange(len(x_out.batch), device=x_out.batch.device),
                x_out.batch,
            )
            s: SelectOutput = self.select.forward(x_out.weights, batch=batch_idx)
            assert s.weight is not None
            x_out = Mixture(
                x_out.positions[s.node_index],
                x_out.weights[s.node_index] * s.weight[:, None],
                batch_idx[s.node_index]
                .histc(len(x_out.batch), min=0, max=len(x_out.batch) - 1)
                .long(),
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
        self.mlp_self = gnn.MLP(
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


class KNN(SparseBase):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        n_layers: int = 2,
        n_samples: int = 1000,
        n_hidden: int = 32,
        n_hidden_mlp: int = 128,
        sigma: float = 10.0,
        max_filter_kernels: int = 25,
        update_positions: bool = False,
        sample_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_layers = n_layers
        self.sigma = sigma
        self.n_samples = n_samples
        self.nonlinearity = nn.LeakyReLU()

        in_channels = self.input_length
        out_channels = 2  # sigma, logp

        self.readin = gnn.MLP(
            [in_channels, n_hidden_mlp, n_hidden],
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=False,
        )
        self.readout = gnn.MLP(
            [n_hidden, n_hidden_mlp, out_channels],
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=True,
        )
        self.convolutions = nn.Sequential(
            *[
                KernelEncoderLayer(
                    max_filter_kernels,
                    n_hidden,
                    n_hidden_mlp,
                    sigma,
                    update_positions,
                    sample_ratio,
                )
                for _ in range(n_layers)
            ]
        )

    @torch.no_grad()
    def forward_input(self, data: SparseData) -> Mixture:
        batch_size = len(data.target_batch_sizes)
        split_idx = data.measurement_batch_sizes.cpu().cumsum(0)[:-1]
        measurement_split = data.measurement_position.tensor_split(split_idx)
        time_split = data.measurement_time.tensor_split(split_idx)
        sensor_split = data.sensor_position.tensor_split(split_idx)

        ## SAMPLING PROCEDURE FOR POSITIONS
        # For each batch we sample n_samples positions
        # 1. Find a random measurement in the final time step.
        # 2. Add uniform noise to the position of the measurement.
        # 3. Repeat n_samples times.
        # TODO: Figure out how to do this in parallel.

        positions = torch.zeros(batch_size, self.n_samples, 2, device=self.device)
        weights = torch.zeros(
            batch_size, self.n_samples, self.input_length, device=self.device
        )
        for batch in range(batch_size):
            time = time_split[batch]
            measurement = measurement_split[batch]
            sensor = sensor_split[batch]

            # get the last time step with at least one measurement
            for i in range(self.input_length):
                last_measurement = measurement[time == (self.input_length - 1 - i)]
                if len(last_measurement) > 0:
                    break

            # increase the number of positions by a factor of 10 by adding noise
            # positions = data.target_position[data.target_time == (input_length - 1)]
            pos_idx = torch.randint(
                0, len(last_measurement), (self.n_samples,), device=self.device
            )
            pos_noise = (
                2 * torch.rand(self.n_samples, 2, device=self.device) - 1.0
            ) * 50
            positions[batch] = last_measurement[pos_idx] + pos_noise

            # sample measurement intensity for each time step
            for t in range(self.input_length):
                # mask for the current time step, not the last one
                time_mask = [time == t]
                weights[batch, :, t] = sample_measurement_intensity(
                    positions[batch], measurement[time_mask], sensor[time_mask]
                )

        # positions.shape = (batch_size, n_samples, 2)
        # weights.shape = (batch_size, n_samples, input_length)
        kernel = GaussianKernel(self.sigma)
        mixture = Mixture(positions, weights)
        mixture = solve_kernel(kernel, mixture, alpha=0.001)
        # combine the batch dimension with the sample dimension
        mixture = Mixture(
            positions=mixture.positions.view(-1, 2),
            weights=mixture.weights.view(-1, self.input_length),
            batch=torch.full(
                (batch_size,), self.n_samples, dtype=torch.long, device=self.device
            ),
        )
        return mixture

    def forward(self, x: Mixture):
        x = x.map_weights(self.readin.forward)
        x = self.convolutions(x)
        x = x.map_weights(self.readout.forward)

        # interpret the output as a mixture of Gaussians
        mu = x.positions
        sigma = x.weights[:, 0, None].expand(-1, 2)
        logits = x.weights[:, 1]
        batch = x.batch
        assert batch is not None
        return self.forward_output(mu, sigma, logits, batch)


def sample_measurement_intensity(
    XY: torch.Tensor,
    measurements: torch.Tensor,
    sensors: torch.Tensor,
    noise_range: float = 10.0,
    noise_bearing: float = 0.035,
) -> torch.Tensor:
    """
    Args:
        XY: (N,2) The positions of the samples.
        measurements: (M, 2) The positions of the measurements.
        sensors: (M, 2) The positions of the sensors.
        noise_range: The standard deviation of the range noise.
        noise_bearing: The standard deviation of the bearing noise.
    """
    XY = XY[:, None, :]
    measurements = measurements[None, :, :]
    sensors = sensors[None, :, :]

    rtheta_sample = to_polar_torch(XY - sensors)
    rtheta_measurement = to_polar_torch(measurements - sensors)
    delta_r = torch.abs(rtheta_sample[..., 0] - rtheta_measurement[..., 0])
    delta_theta = rtheta_sample[..., 1] - rtheta_measurement[..., 1]
    delta_theta = (delta_theta + torch.pi) % (2 * torch.pi) - torch.pi
    Z = (
        torch.exp(
            -0.5 * (delta_r**2 / noise_range**2 + delta_theta**2 / noise_bearing**2)
        )
        / ((2 * torch.pi) ** 0.5 * noise_range * noise_bearing)
        / rtheta_sample[..., 0]
    ).sum(-1)
    return Z

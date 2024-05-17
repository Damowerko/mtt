import math
from typing import NamedTuple

import pykeops.torch as keops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_cluster import grid_cluster
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.nn.pool.select import SelectOutput, SelectTopK
from torchcps.kernel.nn import (
    GaussianKernel,
    Kernel,
    KernelConv,
    KernelSample,
    Mixture,
    sample_kernel,
    solve_kernel,
)

from mtt.data.sparse import SparseData
from mtt.models.sparse import SparseLabel
from mtt.peaks import GMM, reweigh
from mtt.utils import add_model_specific_args, compute_ospa, to_polar_torch


class Select(nn.Module):
    def __init__(self, n_channels: int, ratio: float):
        super().__init__()
        self.select = SelectTopK(n_channels, ratio)

    def forward(self, x: Mixture):
        # see: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.TopKPooling.html
        # we have to multiply the output by the score to make this differentiable
        assert x.batch is not None
        batch_idx = torch.repeat_interleave(
            torch.arange(len(x.batch), device=x.batch.device),
            x.batch,
        )
        s: SelectOutput = self.select.forward(x.weights, batch=batch_idx)
        assert s.weight is not None
        x = Mixture(
            x.positions[s.node_index],
            x.weights[s.node_index] * s.weight[:, None],
            batch_idx[s.node_index]
            .histc(len(x.batch), min=0, max=len(x.batch) - 1)
            .long(),
        )


class Upsample(nn.Module):
    def __init__(self, kernel: Kernel, ratio: float, sigma: float):
        """
        Args:
            kernel: The kernel to use for the upsampling.
            ratio: The ratio of new samples to old samples.
            sigma: The standard deviation of the noise to add to positions.
        """

        super().__init__()
        self.kernel = kernel
        if ratio <= 1.0:
            raise ValueError("Upsampling ratio must be greater than 1.")
        self.ratio = ratio
        self.sigma = sigma

    def forward(self, x: Mixture):
        assert x.batch is not None
        # number of original and new samples
        batch_old = x.batch
        batch_new = (x.batch * self.ratio).floor().long()
        split_old = x.batch.cpu().cumsum(0)
        split_new = x.batch.cpu().cumsum(0)
        positions = torch.zeros(
            int(batch_old.sum() + batch_new.sum()), device=x.positions.device
        )
        for i in range(len(x.batch)):
            # [       OLD                 |                NEW  ]
            # split_new[i] | split_new[i] + n_old[i] | split_new[i+1]
            slice_old = slice(split_new[i], split_new[i] + batch_old[i])
            slice_new = slice(split_new[i] + batch_old[i], split_new[i + 1])

            # copy old positions to position in the new array
            positions_old = x.positions[split_old[i] : split_old[i + 1]]
            positions[slice_old] = positions_old

            # use randint to sample new positions
            idx = torch.randint(
                int(batch_old[i]),
                int(batch_old[i + 1]),
                (int(batch_new[i] - batch_old[i]),),
                device=x.batch.device,
            )
            positions[slice_new] = positions_old[idx]
            positions[slice_new] += torch.randn_like(positions[slice_new]) * self.sigma
        return sample_kernel(self.kernel, x, positions, batch_new)


class SamplingNormalization(nn.Module):
    def __init__(self, kernel: Kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: Mixture):
        K = self.kernel(x.positions, x.positions, x.batch, x.batch)
        return Mixture(
            x.positions,
            x.weights / K.sum(-1),
        )


class KernelEncoderLayer(nn.Module):
    def __init__(
        self,
        max_filter_kernels: int,
        n_channels: int,
        n_channels_mlp: int,
        sigma: float,
        update_positions: bool,
        select_ratio: float = 1.0,
        alpha: float = 0,
        deformable: bool = False,
        sampling_normalization: bool = False,
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
        self.kernel = GaussianKernel(sigma)
        self.sample = KernelSample(
            kernel=self.kernel,
            nonlinearity=nn.LeakyReLU(),
            alpha=alpha if alpha > 0 else None,
        )
        self.conv_norm = nn.BatchNorm1d(n_channels)
        self.mlp = gnn.MLP(
            [
                n_channels,
                n_channels_mlp,
                n_channels + pos_dim if deformable else n_channels,
            ],
            act=nn.LeakyReLU(),
            norm="batch_norm",
            act_first=True,
            plain_last=True,
        )
        self.nonlinearity = nn.LeakyReLU()
        self.select = (
            SelectTopK(n_channels, select_ratio) if select_ratio < 0.99 else None
        )
        self.deformable = deformable
        self.sampling_normalization = (
            SamplingNormalization(self.kernel) if sampling_normalization else None
        )

    def forward(self, x_in: Mixture):
        x = self.conv.forward(x_in)
        # KernelSample applies the nonlinearity after sampling
        x = self.sample.forward(x, x_in.positions, x_in.batch)
        if self.sampling_normalization:
            x = self.sampling_normalization(x)
        x = x.map_weights(self.conv_norm)
        # residual connection
        x = x.map_weights(x_in.weights.add)
        mlp_out = self.mlp.forward(x.weights)
        delta_weights = mlp_out[..., : x.weights.shape[-1]]
        # residual connection
        x_out = Mixture(
            x.positions,
            x.weights + delta_weights,
            x.batch,
        )
        if self.deformable:
            # we perturb the positions by the output of the mlp
            delta_positions = mlp_out[..., x.weights.shape[-1] :]
            x_out = Mixture(
                x_out.positions + delta_positions,
                x_out.weights,
                x_out.batch,
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


def kernel_loss_keops(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, kernel: Kernel
):
    """
    Assumes that x and y are centers of gaussian pulses with variance sigma^2. Computes the MSE loss between the two.
    ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    ArgsL
        x: (N, d) Predicted positions.
        w: (N, d) Weights.
        y: (M, d) Ground truth positions.
    """
    if x.shape[0] == 0:
        xx = 0
    else:
        K_xx = kernel(x, x)
        xx = (w.mT @ (K_xx @ w)).squeeze()

    if y.shape[0] == 0:
        yy = 0
    else:
        K_yy = kernel(y, y)
        # assuming that the ground truth is always 1
        yy = (K_yy @ torch.ones(K_yy.shape[0], device=y.device)).sum()

    if x.shape[0] == 0 or y.shape[0] == 0:
        xy = 0
    else:
        K_yx = kernel(y, x)
        xy = (K_yx @ w).sum()

    loss = xx + yy - 2 * xy
    return loss


def kernel_loss(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, sigma: float):
    """
    Assumes that x and y are centers of gaussian pulses with variance sigma^2. Computes the MSE loss between the two.
    ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    ArgsL
        x: (N, d) Predicted positions.
        w: (N, d) Weights.
        y: (M, d) Ground truth positions.
    """

    pos_dims = x.shape[1]
    x_dist = torch.cdist(x, x, p=2)
    K_xx = torch.exp(-(x_dist**2) / (2 * sigma**2)) / (
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    xx = (w.mT @ K_xx @ w).squeeze()

    y_dist = torch.cdist(y, y, p=2)
    K_yy = torch.exp(-(y_dist**2) / (2 * sigma**2)) / (
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    # assuming that the ground truth is always 1
    yy = K_yy.sum()

    cross_dist = torch.cdist(x, y, p=2)
    K_xy = torch.exp(-(cross_dist**2) / (2 * sigma**2)) / (
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    xy = (w.mT @ K_xy).sum()

    loss = xx + yy - 2 * xy
    return loss


def scaled_logsumexp(x: torch.Tensor, dims: int | tuple[int, ...] = -1):
    """
    Compute the logsumexp of x with scaling.
    """
    if isinstance(dims, int):
        dims = (dims,)
    max_x = x
    for dim in dims:
        max_x, _ = torch.max(max_x, dim=dim, keepdim=True)
    return max_x.squeeze(dims) + torch.logsumexp(x - max_x, dim=dims)


def log_kernel_loss(x: torch.Tensor, logw: torch.Tensor, y: torch.Tensor, sigma: float):
    """
    Same as `kernel` loss but in log-space. Assumes that x and y are centers of gaussian pulses with variance sigma^2. Computes the MSE loss between the two in log-space.
    ||x-y||^2 = logsumexp( log(||x||^2) + log(||y||^2) - log(2<x,y>) )

    Args:
        x: (N, d) Predicted states.
        logw: (N,) Weights in log-space.
        y: (M, d) Ground truth states.
        sigma: float The variance of the gaussian kernel to use.
    """

    pos_dims = x.shape[1]
    x_dist = torch.cdist(x, x, p=2)
    K_xx = (-(x_dist**2) / (2 * sigma**2)) - math.log(
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    xx = scaled_logsumexp(logw[:, None] + K_xx + logw[None, :], dims=(-1, -2))

    y_dist = torch.cdist(y, y, p=2)
    K_yy = (-(y_dist**2) / (2 * sigma**2)) - math.log(
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    # assuming that the ground truth is always 1
    yy = scaled_logsumexp(K_yy, dims=(-1, -2))

    cross_dist = torch.cdist(x, y, p=2)
    K_xy = (-(cross_dist**2) / (2 * sigma**2)) - math.log(
        (2 * math.pi * sigma**2) ** (pos_dims / 2)
    )
    xy = scaled_logsumexp(logw[:, None] + K_xy, dims=(-1, -2))

    loss = torch.logsumexp(torch.stack((xx, yy, -2 * xy), dim=-1), dim=-1)
    return loss


class RKHSBase(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ospa_cutoff: float = 500.0,
        input_length: int = 10,
        kernel_sigma: float = 10.0,
        n_samples: int = 1000,
        cardinality_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.ospa_cutoff = ospa_cutoff
        self.input_length = input_length
        self.kernel_sigma = kernel_sigma
        self.n_samples = n_samples
        self.cardinality_weight = cardinality_weight

    def forward(self, batch: Mixture) -> Mixture:
        raise NotImplementedError("Implement forward method.")

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
        kernel = GaussianKernel(self.sigma[0])
        mixture = Mixture(positions, weights)
        mixture = solve_kernel(kernel, mixture, alpha=0.1)
        # combine the batch dimension with the sample dimension
        mixture = Mixture(
            positions=mixture.positions.view(-1, 2),
            weights=mixture.weights.view(-1, self.input_length),
            batch=torch.full(
                (batch_size,), self.n_samples, dtype=torch.long, device=self.device
            ),
        )
        return mixture

    def training_step(self, batch: SparseData, *_):
        output = self.forward(self.forward_input(batch))
        assert output.batch is not None
        batch_size = output.batch.shape[0]
        label = SparseLabel.from_sparse_data(batch, self.input_length)
        function_mse, cardinality_mse = self.loss(label, output)
        loss = function_mse + cardinality_mse * self.cardinality_weight
        self.log("train/function_mse", function_mse, batch_size=batch_size)
        self.log("train/cardinality_mse", cardinality_mse, batch_size=batch_size)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: SparseData, *_):
        output = self.forward(self.forward_input(batch))
        assert output.batch is not None
        batch_size = output.batch.shape[0]
        label = SparseLabel.from_sparse_data(batch, self.input_length)
        rkhs_mse, cardinality_mse = self.loss(label, output)
        loss = rkhs_mse + cardinality_mse * self.cardinality_weight
        ospa = self.ospa(label, output)
        self.log("val/function_mse", rkhs_mse, batch_size=batch_size)
        self.log("val/cardinality_mse", cardinality_mse, batch_size=batch_size)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/ospa", ospa, prog_bar=True, batch_size=batch_size)
        assert output.batch is not None
        self.log(
            "val/n_outputs",
            output.positions.shape[0] / batch_size,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def loss(
        self,
        label: SparseLabel,
        output: Mixture,
    ):
        assert output.batch is not None
        x_split_idx = output.batch.cumsum(0)[:-1].cpu()
        y_split_idx = label.batch.cumsum(0)[:-1].cpu()

        pos_split = output.positions.tensor_split(x_split_idx)
        weights_split = output.weights.tensor_split(x_split_idx)
        y_split = label.y.tensor_split(y_split_idx)

        batch_size = output.batch.shape[0]
        rkhs_mse = torch.zeros((batch_size,), device=self.device)
        for batch_idx in range(batch_size):
            rkhs_mse[batch_idx] = kernel_loss(
                pos_split[batch_idx],
                weights_split[batch_idx][..., :1],
                y_split[batch_idx],
                self.kernel_sigma,
            )

        cardinality_mse = torch.nn.functional.mse_loss(
            self.estimate_cardinality(output).view(batch_size),
            label.batch.float(),
        )
        return rkhs_mse.mean(), cardinality_mse

    def estimate_cardinality(self, output: Mixture):
        weights = output.weights[..., -1]
        if output.batch is None:
            return weights.sum()
        else:
            split_idx = output.batch.cpu().cumsum(0)[:-1]
            weights_split = weights.tensor_split(split_idx)
            return torch.stack([w.sum() for w in weights_split])

    def ospa(self, label: SparseLabel, output: Mixture):
        assert output.batch is not None
        x_split_idx = output.batch.cumsum(0)[:-1].cpu()
        y_split_idx = label.batch.cumsum(0)[:-1].cpu()

        pos_split = output.positions.tensor_split(x_split_idx)
        weights_split = output.weights.tensor_split(x_split_idx)
        y_split = label.y.tensor_split(y_split_idx)

        ospa = torch.zeros((output.batch.shape[0],), device=self.device)
        for batch_idx in range(output.batch.shape[0]):
            X = self.find_peaks(Mixture(pos_split[batch_idx], weights_split[batch_idx]))
            Y = y_split[batch_idx].detach().cpu().numpy()
            ospa[batch_idx] = compute_ospa(X, Y, self.ospa_cutoff, p=2)
        return ospa.mean()

    def find_peaks(self, output: Mixture):
        kernel = GaussianKernel(self.kernel_sigma * 2)
        pos = output.positions.detach()
        weights = output.weights.detach()[..., :1].squeeze().contiguous()
        X = pos.clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([X], lr=10, weight_decay=0.0)
        with torch.enable_grad():
            for _ in range(100):
                # the evaluation of the RKHS at pos is the "likelyhood"
                likelyhood = kernel(X, pos) @ weights
                optimizer.zero_grad()
                (-likelyhood.sum()).backward()
                optimizer.step()
        X = X.detach()
        cluster = grid_cluster(X, torch.full((X.shape[-1],), 10.0, device=X.device))
        X = avg_pool_x(
            cluster, X, torch.full(X.shape[:1], 1, device=X.device, dtype=torch.long)
        )[0]
        n_components = self.estimate_cardinality(output).item()
        likelyhood = kernel(X, pos) @ weights
        mu = X.cpu().numpy()
        gmm = reweigh(GMM(mu, mu, likelyhood.cpu().numpy()), n_components)
        return gmm.means

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class KNN(RKHSBase):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        n_layers: int = 2,
        n_hidden: int = 32,
        n_hidden_mlp: int = 128,
        sigma: list[float] = [10.0],
        max_filter_kernels: int = 25,
        update_positions: bool = False,
        sample_ratio: float = 1.0,
        alpha: float = 0,
        deformable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if isinstance(sigma, float):
            self.sigma = [sigma] * n_layers
        else:
            self.sigma = sigma * n_layers if len(sigma) == 1 else sigma
        self.n_layers = n_layers
        self.nonlinearity = nn.LeakyReLU()

        in_channels = self.input_length
        out_channels = 2  # weights

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
                    self.sigma[l],
                    update_positions,
                    sample_ratio,
                    alpha=alpha,
                    deformable=deformable,
                )
                for l in range(n_layers)
            ]
        )

    def forward(self, x: Mixture) -> Mixture:
        x = x.map_weights(self.readin.forward)
        x = self.convolutions(x)
        x = x.map_weights(self.readout.forward)
        x = x.map_weights(self.nonlinearity)
        return x

    def configure_optimizers(self):
        # get all parameters that are KernelConv.kernel_positions
        position_parameters = [
            parameter
            for name, parameter in self.named_parameters()
            if "kernel_positions" in name
        ]
        # all other parameters
        other_parameters = [
            parameter
            for name, parameter in self.named_parameters()
            if "kernel_positions" not in name
        ]
        return torch.optim.AdamW(
            [
                {
                    "params": other_parameters,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                # we don't want regularization on positions
                # also learning rate is 10 times smaller
                {
                    "params": position_parameters,
                    "lr": self.learning_rate * 0.1,
                    "weight_decay": 0.0,
                },
            ]
        )


@torch.no_grad()
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

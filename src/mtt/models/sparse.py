import math
import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mtt.data.sparse import SparseData
from mtt.utils import add_model_specific_args, compute_ospa


class SparseInput(NamedTuple):
    x: torch.Tensor
    x_pos: torch.Tensor
    x_batch: torch.Tensor

    @staticmethod
    def from_sparse_data(data: SparseData, input_length: int):
        """
        The input to the model is a tuple of (x, x_pos, x_batch).
        This is how I construct STInput:
            x: The sensor positions relative to the measurement positions.
            x_pos: The measurement positions.
            x_batch: The batch vector. For now lets assume a batch size of 1.
        """
        time = data.measurement_time
        sensor_pos = data.sensor_position - data.measurement_position
        x = torch.zeros(
            (
                sensor_pos.shape[0],
                sensor_pos.shape[1] * input_length,
            ),
            device=sensor_pos.device,
            dtype=sensor_pos.dtype,
        )
        for i in range(sensor_pos.shape[1]):
            x[:, 2 * time + i] = sensor_pos[:, i]

        x_pos = data.measurement_position
        x_batch = data.measurement_batch_sizes
        return SparseInput(x, x_pos, x_batch)


class SparseLabel(NamedTuple):
    y: torch.Tensor
    batch: torch.Tensor

    @staticmethod
    def from_sparse_data(data: SparseData, input_length) -> "SparseLabel":
        # only keep the last time step
        mask = data.target_time == input_length - 1
        y = data.target_position[mask]

        # to compute the number of elements in each batch after the mast
        # first convert to a batch index, mask, and then count the number of elements
        batch_idx = torch.repeat_interleave(
            torch.arange(
                data.target_batch_sizes.shape[0], device=data.target_batch_sizes.device
            ),
            data.target_batch_sizes,
        )
        y_batch = batch_idx[mask].bincount(minlength=data.target_batch_sizes.shape[0])
        assert y_batch.shape == data.target_batch_sizes.shape
        return SparseLabel(y, y_batch)


class SparseOutput(NamedTuple):
    mu: torch.Tensor
    sigma: torch.Tensor
    logp: torch.Tensor
    batch: torch.Tensor


def logp_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    existance_logp: torch.Tensor,
    y: torch.Tensor,
    assignment: tuple[torch.Tensor, torch.Tensor],
):
    """
    Returns the negative log-likelihood of the assignment between mu and y.
    """

    i, j = assignment
    dist = torch.distributions.Normal(mu[i], sigma[i])
    logp = torch.sum(existance_logp[i] + dist.log_prob(y[j]).sum(-1))

    # add back the (1-p) of the ignored compotents
    # first get a mask for things not in the topk
    mask = torch.ones_like(existance_logp, dtype=torch.bool)
    mask[i] = False
    complement = 1 - existance_logp[mask].exp()
    logp += complement.clamp(min=1e-16, max=(1 - 1e-16)).log().sum()
    return -logp


def mse_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    assignment: tuple[torch.Tensor, torch.Tensor],
):
    """
    Compute the mean squared error between the assignment between x and y.
    """
    i, j = assignment
    loss = (x[i] - y[j]).pow(2).mean()
    return loss


def parallel_assignment(
    x: list[torch.Tensor], y: list[torch.Tensor], logp: list[torch.Tensor] | None = None
) -> typing.Generator[tuple[int, torch.Tensor, torch.Tensor], None, None]:
    """
    Uses a ThreadPoolExecutor to parallelize the assignment problem for each batch.

    Args:
        x: A list of tensors of shape (N, d) representing the predicted states.
        y: A list of tensors of shape (M, d) representing the ground truth states.
        logp (optional): A list of tensors of shape (N,) representing the existence probabilities in log-space.

    Returns:
        A generator that yields the batch index and the assignment for each batch.
            batch_idx: The index of the batch.
            i
    """
    batch_size = len(x)
    if logp is None:
        logp = [torch.zeros_like(_x[:, 0]) for _x in x]
    futures = {}
    with ThreadPoolExecutor() as e, torch.no_grad():
        # map (i,j) -> batch_idx
        for batch_idx in range(batch_size):
            # find a matching between mu_i and y_j
            dist = torch.cdist(x[batch_idx], y[batch_idx], p=2)
            match_cost = dist - logp[batch_idx][:, None]
            future = e.submit(linear_sum_assignment, match_cost.cpu().numpy())
            futures[future] = batch_idx

    for future in as_completed(futures.keys()):
        batch_idx = futures[future]
        i, j = future.result()
        yield batch_idx, i, j


class SparseBase(pl.LightningModule, ABC):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ospa_cutoff: float = 500.0,
        input_length: int = 4,
        loss_type: str = "logp",
        kernel_sigma: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the Sparse model.

        Args:
            lr (float): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float): The weight decay for the optimizer. Defaults to 0.0.
            ospa_cutoff (float): The cutoff value for the OSPA metric. Defaults to 500.0.
            input_length (int): The length of the input sequence. Defaults to 20.
            loss_type (str): The type of loss to use. Either "logp" or "kernel" or "mse".
            **kwargs: Additional keyword arguments. These are ignored.

        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.ospa_cutoff = ospa_cutoff
        self.input_length = input_length
        self.loss_type = loss_type
        self.kernel_sigma = kernel_sigma

    @abstractmethod
    def forward_input(self, data: SparseData) -> typing.Any:
        """
        Parse SparseData to the input format expected by the model.
        """
        pass

    @abstractmethod
    def forward(self, data: typing.Any) -> SparseOutput:
        """
        Takes the input in the format returned by `forward_input` and returns `SparseOutput`.
        """
        pass

    def forward_output(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        logits: torch.Tensor,
        batch: torch.Tensor,
    ) -> SparseOutput:
        sigma = F.softplus(sigma) + 1e-16
        logp = F.logsigmoid(logits)
        if self.loss_type == "kernel":
            logp = logits
        return SparseOutput(mu, sigma, logp, batch)

    def logp_loss(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        existance_logp: torch.Tensor,
        x_batch_sizes: torch.Tensor,
        y: torch.Tensor,
        y_batch_sizes: torch.Tensor,
    ):
        """
        Evaluate the approximate log-likelyhood of the MB following [1].

        [1] J. Pinto, G. Hess, W. Ljungbergh, Y. Xia, H. Wymeersch, and L. Svensson, “Can Deep Learning be Applied to Model-Based Multi-Object Tracking?” arXiv, Feb. 16, 2022. doi: 10.48550/arXiv.2202.07909.

        Args:
            mu: (N, d) Predicted states.
            cov: (N, d) Covariance of the states.
            logp: (N,) Existence probabilities if log-space.
            mu_batch: (A,) The size of each batch in mu, cov, and logp.
            y: (M, d) Ground truth states.
            y_batch: (B,) The size of each batch in y.
        Returns:
            logp: Approximate log-likelyhood of the MB given the ground truth. NB: Takes average over the batch.
        """
        batch_size = x_batch_sizes.shape[0]
        # tensor_split expects indices at which to split tensor
        # they must be on the CPU
        x_split_idx = x_batch_sizes.cumsum(0)[:-1].cpu()
        y_split_idx = y_batch_sizes.cumsum(0)[:-1].cpu()

        mu_split = mu.tensor_split(x_split_idx)
        sigma_split = sigma.tensor_split(x_split_idx)
        logp_split = existance_logp.tensor_split(x_split_idx)
        y_split = y.tensor_split(y_split_idx)

        loss = torch.zeros((batch_size,), device=self.device)
        if self.loss_type == "logp":
            for batch_idx, i, j in parallel_assignment(mu_split, y_split, logp_split):
                loss[batch_idx] = logp_loss(
                    mu_split[batch_idx],
                    sigma_split[batch_idx],
                    logp_split[batch_idx],
                    y_split[batch_idx],
                    (i, j),
                )

        position_loss = torch.zeros((batch_size,), device=self.device)
        with ThreadPoolExecutor() as e, torch.no_grad():
            # map (i,j) -> batch_idx
            futures = {}
            for batch_idx in range(batch_size):
                # find a matching between mu_i and y_j
                dist = torch.cdist(mu_split[batch_idx], y_split[batch_idx], p=2)
                match_cost = dist - logp_split[batch_idx][:, None]
                future = e.submit(linear_sum_assignment, match_cost.cpu().numpy())
                futures[future] = batch_idx

        logp = torch.zeros((batch_size,), device=self.device)
        for future in as_completed(futures.keys()):
            batch_idx = futures[future]
            i, j = future.result()

            _mu = mu_split[batch_idx]
            _sigma = sigma_split[batch_idx]
            _existance_logp = logp_split[batch_idx]
            _y = y_split[batch_idx]

            assert _mu.shape[0] == x_batch_sizes[batch_idx]
            assert _y.shape[0] == y_batch_sizes[batch_idx]
            assert i.shape[0] == min(_mu.shape[0], _y.shape[0])

            dist = torch.distributions.Normal(_mu[i], _sigma[i])
            logp[batch_idx] = torch.sum(
                _existance_logp[i] + dist.log_prob(_y[j]).sum(-1)
            )

            # add back the (1-p) of the ignored compotents
            # first get a mask for things not in the topk
            mask = torch.ones_like(_existance_logp, dtype=torch.bool)
            mask[i] = False
            complement = 1 - _existance_logp[mask].exp()
            logp[batch_idx] += complement.clamp(min=1e-16, max=(1 - 1e-16)).log().sum()

        # average probability across batches
        if batch_size < 1:
            print(f"Warning: batch size is {batch_size} for some reason.")
        else:
            logp = torch.logsumexp(logp, 0) - math.log(batch_size)
            position_loss = position_loss.mean()

        if self.position_loss:
            self.log("train/position_loss", position_loss, prog_bar=True)

        loss = logp + position_loss
        return loss

    def loss(
        self,
        label: SparseLabel,
        output: SparseOutput,
        loss_type: str,
    ):
        """
        Compute the loss between the label and the output. There are several types of loss functions that can be used.

        Args:
            label: The ground truth label.
            output: The model output.
            loss_type: The type of loss to use. Either "logp" or "kernel" or "mse" or "logp_kernel".
                - "logp": The negative log-likelihood of the assignment between mu and y.
                - "mse": (OSPA) The mean squared error between the optimal assignment between mu and y.

        """

        # tensor_split expects indices at which to split tensor
        # they must be on the CPU
        x_split_idx = output.batch.cumsum(0)[:-1].cpu()
        y_split_idx = label.batch.cumsum(0)[:-1].cpu()
        mu_split = output.mu.tensor_split(x_split_idx)
        sigma_split = output.sigma.tensor_split(x_split_idx)
        logp_split = output.logp.tensor_split(x_split_idx)
        y_split = label.y.tensor_split(y_split_idx)

        batch_size = output.batch.shape[0]
        loss = torch.zeros((batch_size,), device=self.device)

        if loss_type == "logp":
            for batch_idx, i, j in parallel_assignment(mu_split, y_split, logp_split):
                if len(mu_split) == 0:
                    continue
                loss[batch_idx] = logp_loss(
                    mu_split[batch_idx],
                    sigma_split[batch_idx],
                    logp_split[batch_idx],
                    y_split[batch_idx],
                    (i, j),
                )
        elif loss_type == "mse":
            for batch_idx, i, j in parallel_assignment(mu_split, y_split):
                if len(mu_split) == 0:
                    continue
                loss[batch_idx] = mse_loss(
                    mu_split[batch_idx], y_split[batch_idx], (i, j)
                )
        return loss.mean()

    def ospa(self, label: SparseLabel, output: SparseOutput):
        x_split_idx = output.batch.cumsum(0)[:-1].cpu()
        y_split_idx = label.batch.cumsum(0)[:-1].cpu()
        mu_split = output.mu.tensor_split(x_split_idx)
        logp_split = output.logp.tensor_split(x_split_idx)
        y_split = label.y.tensor_split(y_split_idx)

        ospa = torch.zeros((output.batch.shape[0],), device=self.device)
        for batch_idx in range(output.batch.shape[0]):
            mu = mu_split[batch_idx]
            logp = logp_split[batch_idx]
            X = mu[logp.exp() > 0.5].detach().cpu().numpy()
            Y = y_split[batch_idx].detach().cpu().numpy()
            ospa[batch_idx] = compute_ospa(X, Y, self.ospa_cutoff, p=2)
        return ospa.mean()

    def training_step(self, data: SparseData, *_):
        output = self.forward(self.forward_input(data))
        label = SparseLabel.from_sparse_data(data, self.input_length)
        loss = self.loss(label, output, self.loss_type)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data: SparseData, *_):
        output = self.forward(self.forward_input(data))
        label = SparseLabel.from_sparse_data(data, self.input_length)
        loss = self.loss(label, output, self.loss_type)

        batch_size = output.batch.shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        ospa = self.ospa(label, output)
        self.log("val/ospa", ospa, prog_bar=True, batch_size=batch_size)
        self.log(
            "val/n_outputs", output.mu.shape[0] / batch_size, batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

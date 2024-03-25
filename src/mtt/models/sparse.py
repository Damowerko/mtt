import math
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


class SparseOutput(NamedTuple):
    mu: torch.Tensor
    sigma: torch.Tensor
    logp: torch.Tensor
    batch: torch.Tensor


class SparseLabel(NamedTuple):
    y: torch.Tensor
    y_batch: torch.Tensor


class SparseBase(pl.LightningModule, ABC):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ospa_cutoff: float = 500.0,
        input_length: int = 1,
        **kwargs,
    ):
        """
        Initialize the Sparse model.

        Args:
            lr (float): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float): The weight decay for the optimizer. Defaults to 0.0.
            ospa_cutoff (float): The cutoff value for the OSPA metric. Defaults to 500.0.
            input_length (int): The length of the input sequence. Defaults to 20.
            **kwargs: Additional keyword arguments. These are ignored.

        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.ospa_cutoff = ospa_cutoff
        self.input_length = input_length

    def to_stinput(self, data: SparseData) -> SparseInput:
        """
        The input to the model is a tuple of (x, x_pos, x_batch).
        This is how I construct STInput:
            x: The sensor positions relative to the measurement positions.
            x_pos: The measurement positions.
            x_batch: The batch vector. For now lets assume a batch size of 1.
        """
        x = torch.cat(
            (
                # Sensor positions are (S, d) and sensor_index is (N,)
                # make the sensor positions relative to the measurement positions
                data.sensor_position - data.measurement_position,
                # add the temporal index as a feature
                data.measurement_time[:, None],
            ),
            dim=1,
        )
        x_pos = data.measurement_position
        x_batch = data.measurement_batch_sizes
        return SparseInput(x, x_pos, x_batch)

    def to_stlabel(self, data: SparseData) -> SparseLabel:
        # only keep the last time step
        mask = data.target_time == self.input_length - 1
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

    def forward(
        self, x: torch.Tensor, x_pos: torch.Tensor, x_batch: torch.Tensor
    ) -> SparseOutput:
        # apply activation to the output of _forward
        mu, sigma, logits, batch = self._forward(x, x_pos, x_batch)
        # Sigma must be > 0.0 for torch.distributions.Normal
        sigma = F.softplus(sigma) + 1e-16
        logp = F.logsigmoid(logits)
        return SparseOutput(mu, sigma, logp, batch)

    @abstractmethod
    def _forward(
        self, x: torch.Tensor, x_pos: torch.Tensor, x_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partial implementation of the forward pass.
        Should have no activation function on the output.
        The output of this function is used in `self.forward`.

        Returns:
            mu: (N, d) Predicted states.
            sigma: (N, d) Covariance of the states.
            logits: (N,) Existence probabilities in logit space.
            batch: (B,) The size of each batch in mu, cov, and logp.
        """
        raise NotImplementedError()

    def logp(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        existance_logp: torch.Tensor,
        x_batch_sizes: torch.Tensor,
        y: torch.Tensor,
        y_batch_sizes: torch.Tensor,
        return_average_probability: bool = True,
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
            return_average_probability: If True, return the average probability across the batch (in log-space).
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

        with ThreadPoolExecutor() as e:
            # map (i,j) -> batch_idx
            futures = {}
            for batch_idx in range(batch_size):
                # find a matching between mu_i and y_j
                with torch.no_grad():
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
        if return_average_probability:
            if batch_size < 1:
                print(f"Warning: batch size is {batch_size} for some reason.")
            else:
                logp = torch.logsumexp(logp, 0) - math.log(batch_size)
        return logp

    def training_step(self, data: SparseData, *_):
        input = self.to_stinput(data)
        label = self.to_stlabel(data)
        output = self.forward(*input)

        batch_size = input.x_batch.shape[0]

        logp = self.logp(
            output.mu,
            output.sigma,
            output.logp,
            output.batch,
            label.y,
            label.y_batch,
        )
        loss = -logp
        self.log("train/loss", loss, batch_size=batch_size)
        self.log("train/prob", logp.exp(), prog_bar=True, batch_size=batch_size)
        self.log("train/logp", logp, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data: SparseData, *_):
        input = self.to_stinput(data)
        label = self.to_stlabel(data)
        output = self.forward(*input)

        batch_size = input.x_batch.shape[0]

        logp = self.logp(
            output.mu,
            output.sigma,
            output.logp,
            output.batch,
            label.y,
            label.y_batch,
        )

        loss = -logp
        self.log("val/loss", loss, batch_size=batch_size)
        self.log("val/logp", logp, prog_bar=True, batch_size=batch_size)

        X = output.mu[output.logp.exp() > 0.5]
        ospa = compute_ospa(
            X.detach().cpu().numpy(),
            label.y.detach().cpu().numpy(),
            self.ospa_cutoff,
            p=2,
        )
        self.log("val/ospa", ospa, prog_bar=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

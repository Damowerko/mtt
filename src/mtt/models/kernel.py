import pytorch_lightning as pl
import torch
import torchcps.kernel.nn as knn

from mtt.models.sparse import SparseBase, SparseOutput
from mtt.utils import add_model_specific_args


class KNN(SparseBase):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        measurement_dim: int,
        state_dim: int,
        pos_dim: int,
        n_weights: int = 32,
        n_channels: int = 32,
        n_layers: int = 2,
        sigma: float = 10.0,
        max_filter_kernels: int = 100,
        update_positions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = knn.KNN(
            in_weights=measurement_dim,
            out_weights=state_dim + 1,
            n_weights=n_weights,
            n_layers=n_layers,
            n_channels=n_channels,
            sigma=sigma,
            max_filter_kernels=max_filter_kernels,
            update_positions=update_positions,
        )

    def forward(
        self, x: torch.Tensor, x_pos: torch.Tensor, x_batch: torch.Tensor
    ) -> SparseOutput:
        x_mixture = knn.Mixture(x_pos, x)
        y_mixture = self.model.forward(x_mixture, x_batch)
        mu = y_mixture.positions
        sigma = y_mixture.weights[:-1]
        logp = y_mixture.weights[-1]
        return SparseOutput(mu=mu, sigma=sigma, logp=logp)

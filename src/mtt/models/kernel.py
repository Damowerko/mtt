import typing

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
        self.model = knn.KNN(
            pos_dim,
            in_channels=measurement_dim + self.time_dim,
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
        x_mixture = knn.Mixture(x_pos, x, x_batch)
        y_mixture = self.model.forward(x_mixture)
        mu = y_mixture.positions
        sigma = y_mixture.weights[:, :-1]
        logits = y_mixture.weights[:, -1]
        batch = y_mixture.batch
        return mu, sigma, logits, batch

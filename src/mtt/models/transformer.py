import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import bipartite_subgraph, subgraph
from torchcps.attention import SpatialAttention

from mtt.data.sparse import SparseData
from mtt.models.sparse import SparseBase
from mtt.utils import add_model_specific_args, compute_ospa


class STInput(NamedTuple):
    x: torch.Tensor
    x_pos: torch.Tensor
    x_batch: torch.Tensor


class STOutput(NamedTuple):
    mu: torch.Tensor
    sigma: torch.Tensor
    logp: torch.Tensor


class STLabel(NamedTuple):
    y: torch.Tensor
    y_batch: torch.Tensor


class STData(NamedTuple):
    input: STInput
    label: STLabel


class SpatialTransformerEncoder(nn.Module):
    def __init__(
        self, n_channels: int, n_layers: int, pos_dim: int, heads=1, dropout=0.0
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.heads = heads

        self.attention = nn.ModuleList(
            [
                SpatialAttention(
                    n_channels,
                    pos_dim,
                    heads,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.mlp = nn.ModuleList(
            [
                gnn.MLP(
                    [n_channels * heads, n_channels * heads, n_channels * heads],
                    bias=True,
                    dropout=dropout,
                    act="leaky_relu",
                    norm="batch_norm",
                    plain_last=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_att = nn.ModuleList(
            [gnn.BatchNorm(n_channels * heads) for _ in range(n_layers)]
        )
        self.norm_mlp = nn.ModuleList(
            [gnn.BatchNorm(n_channels * heads) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Spatial Transformer Encoder
        Layers:
         - Spatial Self Attention
         - Add & Norm
         - FFN
         - Add & Norm

        Args:
            x: (N, n_channels * heads) Input tensor.
            pos: (N, pos_dim) Positional embeddings.
            edge_index: (2, E) Tensor representing the edges in the graph.
        """
        if x.shape[-1] != self.n_channels * self.heads:
            raise ValueError(
                f"Expected input dimension (..., n_channels * heads) = (..., {self.n_channels * self.heads})  but got (..., {x.shape[-1]})."
            )

        for i in range(self.n_layers):
            y = self.attention[i](x, pos, edge_index)
            x = self.norm_att[i](x + y)
            y = self.mlp[i](x)
            x = self.norm_mlp[i](x + y)
        return x


class SelectionMechanism(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_channels: int,
        n_hidden: int,
        n_layers: int,
        threshold: float,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
            measurement_dim: Dimension of the measurement vector.
            n_channels: Number of input channels.
            n_hidden: Number of hidden channels.
            n_layers: Number of layers.
            threshold: Embeddings with scores below this will be discarded.
            dropout: Dropout rate. Defaults to 0.0.
        """

        super().__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.threshold = threshold

        self.mlp = gnn.MLP(
            in_channels=n_channels,
            hidden_channels=n_hidden,
            out_channels=input_dim + n_channels + 1,
            num_layers=n_layers,
            act="leaky_relu",
            norm="batch_norm",
            bias=True,
            dropout=dropout,
            plain_last=True,
        )
        self.score_norm = gnn.BatchNorm(1)

    def forward(
        self,
        inputs: torch.Tensor,
        embeddings: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Args:
            inputs: (N, input_dim) Input tensor will be updated.
            embeddings: (N, n_channels) Embeddings from the encoder.
            batch: (N,) Batch vector denoting which batch each elements belongs to.
            batch_size: Number of objects in each batch.
        """

        x = self.mlp(embeddings, batch, batch_size)

        update = F.leaky_relu(x[..., : self.input_dim])
        objects = F.leaky_relu(
            x[..., self.input_dim : self.input_dim + self.n_channels]
        )
        score = self.score_norm(x[..., -1, None])[..., 0].sigmoid()

        # Keep only embeddings with score above threshold
        mask = score > self.threshold
        inputs_updated = inputs[mask, :] + update[mask, :]
        objects = objects[mask, :]
        return inputs_updated, objects, mask


class SpatialTransformerDecoder(nn.Module):
    def __init__(
        self, n_channels: int, n_layers: int, pos_dim: int, heads=1, dropout=0.0
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers

        self.self_attention = nn.ModuleList(
            [
                SpatialAttention(
                    n_channels,
                    pos_dim,
                    heads,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.cross_attention = nn.ModuleList(
            [
                SpatialAttention(
                    n_channels,
                    pos_dim,
                    heads,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.mlp = nn.ModuleList(
            [
                gnn.MLP(
                    [n_channels * heads, n_channels * heads, n_channels * heads],
                    bias=True,
                    dropout=dropout,
                    act="leaky_relu",
                    norm="batch_norm",
                    plain_last=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_self = nn.ModuleList(
            [gnn.BatchNorm(n_channels * heads) for _ in range(n_layers)]
        )
        self.norm_cross = nn.ModuleList(
            [gnn.BatchNorm(n_channels * heads) for _ in range(n_layers)]
        )
        self.norm_mlp = nn.ModuleList(
            [gnn.BatchNorm(n_channels * heads) for _ in range(n_layers)]
        )

    def forward(
        self,
        encoding: torch.Tensor,
        object: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        # k x k subgraph
        edge_index_object = subgraph(mask, edge_index, relabel_nodes=True)[0]
        # N x k subgraph
        edge_index_cross = bipartite_subgraph(
            (
                torch.ones_like(mask),
                mask,
            ),
            edge_index,
            relabel_nodes=True,
        )[0]
        pos_masked = pos[mask]
        for i in range(self.n_layers):
            # self attention (k x k) on object
            object_self = self.self_attention[i](object, pos_masked, edge_index_object)
            object_self = self.norm_self[i](object + object_self)
            # cross attention from encoding to object (N x k)
            object_cross = self.cross_attention[i](
                (encoding, object_self), (pos, pos_masked), edge_index_cross
            )
            object_cross = self.norm_cross[i](object_self + object_cross)
            # feed forward with residual connection
            object_mlp = self.mlp[i](object_cross)
            object = self.norm_mlp[i](object_cross + object_mlp)
        return object


class SpatialTransformer(SparseBase):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        measurement_dim: int,
        state_dim: int,
        pos_dim: int,
        n_channels: int = 32,
        n_encoder: int = 2,
        n_decoder: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
        radius: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            measurement_dim=measurement_dim,
            state_dim=state_dim,
            pos_dim=pos_dim,
            **kwargs,
        )
        self.save_hyperparameters()
        self.radius = radius
        self.state_dim = state_dim
        # output channels for mu, sigma and probability
        self.out_channels = 2 * state_dim + 1

        self.in_norm = gnn.BatchNorm(measurement_dim)
        self.readin = gnn.MLP(
            [measurement_dim, n_channels * heads, n_channels * heads],
            bias=True,
            dropout=dropout,
            act="leaky_relu",
            norm="batch_norm",
            plain_last=False,
        )
        self.readout = gnn.MLP(
            [n_channels * heads, n_channels * heads, self.out_channels],
            bias=True,
            dropout=dropout,
            act="leaky_relu",
            norm="batch_norm",
            plain_last=True,
        )
        self.encoder = SpatialTransformerEncoder(
            n_channels, n_encoder, pos_dim, heads, dropout
        )
        self.decoder = SpatialTransformerDecoder(
            n_channels, n_decoder, pos_dim, heads, dropout
        )

    def _forward(
        self,
        x: torch.Tensor,
        x_pos: torch.Tensor,
        x_batch: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (N, in_channels) Non-position measurement data.
            pos: (N, pos_dim) Positions of the measurements.
        Returns:
            states: (N, state_dim) State predictions.
            probabilities: (N,) Existence probabilities.
        """
        # convert batch sizes to indices in {1,...,B}^N
        if x_batch is not None:
            x_batch = torch.repeat_interleave(
                torch.arange(x_batch.shape[0], device=self.device), x_batch
            )
        else:
            x_batch = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)

        # create graph based on positions
        edge_index = gnn.radius_graph(
            x_pos,
            r=self.radius,
            batch=x_batch,
            loop=True,
        ).to(self.device)

        # Normalize input
        x = self.in_norm.forward(x)
        # Readin
        x = self.readin.forward(x, x_batch)
        # Encoder
        encoding = self.encoder.forward(x, x_pos, edge_index)
        # Decoder
        # at the moment we just use all encodings as the object queries
        object = encoding  # TODO: Selection mechanism
        object = self.decoder.forward(
            encoding,
            object,
            x_pos,
            mask=torch.ones((x.shape[0],), dtype=torch.bool, device=self.device),
            edge_index=edge_index,
        )
        # Readout
        object = self.readout.forward(object, x_batch)
        # Split into existence probability and state
        mu = object[..., : self.state_dim]
        sigma = object[..., self.state_dim : 2 * self.state_dim]
        logits = object[..., -1]
        return mu, sigma, logits

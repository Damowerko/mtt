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

from mtt.data.tensor import TensorData
from mtt.utils import add_model_specific_args


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


class SpatialTransformer(pl.LightningModule):
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
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.radius = radius
        self.state_dim = state_dim
        self.lr = lr
        self.weight_decay = weight_decay
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

    def to_stinput(self, data: TensorData) -> STInput:
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
        return STInput(x, x_pos, x_batch)

    def to_stlabel(self, data: TensorData) -> STLabel:
        y = data.target_position
        y_batch = data.target_batch_sizes
        return STLabel(y, y_batch)

    def forward(
        self,
        x: torch.Tensor,
        x_pos: torch.Tensor,
        x_batch: Optional[torch.Tensor] = None,
    ) -> STOutput:
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
        # x = self.in_norm.forward(x)
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
        # Sigma must be > 0.0 for torch.distributions.Normal
        sigma = object[..., self.state_dim : 2 * self.state_dim].abs().clamp(min=1e-16)
        logp = F.logsigmoid(object[..., -1])
        return STOutput(mu, sigma, logp)

    def logp(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        existance_logp: torch.Tensor,
        mu_batch_sizes: torch.Tensor,
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
        Returns:
            logp: Approximate log-likelyhood of the MB given the ground truth. NB: Takes average over the batch.
        """
        # need to be on CPU for tensor_split
        mu_batch_sizes = mu_batch_sizes.cpu()
        y_batch_sizes = y_batch_sizes.cpu()

        batch_size = mu_batch_sizes.shape[0]
        mu_split = mu.tensor_split(mu_batch_sizes)
        sigma_split = sigma.tensor_split(mu_batch_sizes)
        logp_split = existance_logp.tensor_split(mu_batch_sizes)
        y_split = y.tensor_split(y_batch_sizes)

        with ThreadPoolExecutor() as e:
            futures = {}
            for batch_idx in range(batch_size):
                # find a matching between mu_i and y_j
                with torch.no_grad():
                    match_cost = (
                        torch.cdist(mu_split[batch_idx], y_split[batch_idx], p=2)
                        - logp_split[batch_idx][:, None]
                    )
                    future = e.submit(linear_sum_assignment, match_cost.cpu().numpy())
                    futures[future] = batch_idx

        logp = torch.zeros((batch_size,), device=self.device)
        for future in as_completed(futures):
            batch_idx = futures[future]
            i, j = future.result()

            _mu = mu_split[batch_idx]
            _sigma = sigma_split[batch_idx]
            _existance_logp = logp_split[batch_idx]
            _y = y_split[batch_idx]

            dist = torch.distributions.Normal(_mu[i], _sigma[i])
            logp[batch_idx] = torch.sum(
                _existance_logp[i] + dist.log_prob(_y[j]).sum(-1)
            )

            # add back the (1-p) of the ignored compotents
            # first get a mask for things not in the topk
            mask = torch.ones_like(_existance_logp, dtype=torch.bool)
            mask[i] = False
            logp[batch_idx] += torch.log1p(-_existance_logp[mask].exp()).sum()

        # average probability across batches
        if return_average_probability:
            logp = torch.logsumexp(logp, 0) - math.log(batch_size)
        return logp

    def training_step(self, data: TensorData, *_):
        input = self.to_stinput(data)
        label = self.to_stlabel(data)
        output = self.forward(*input)

        batch_size = input.x_batch.shape[0]

        logp = self.logp(
            output.mu,
            output.sigma,
            output.logp,
            input.x_batch,
            label.y,
            label.y_batch,
        )
        loss = -logp
        self.log("train/loss", loss, batch_size=batch_size)
        self.log("train/prob", logp.exp(), prog_bar=True, batch_size=batch_size)
        self.log("train/logp", logp, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data: TensorData, *_):
        input = self.to_stinput(data)
        label = self.to_stlabel(data)
        output = self.forward(*input)

        batch_size = input.x_batch.shape[0]

        logp = self.logp(
            output.mu,
            output.sigma,
            output.logp,
            input.x_batch,
            label.y,
            label.y_batch,
        )

        loss = -logp
        self.log("val/loss", loss, batch_size=batch_size)
        self.log("val/logp", logp, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

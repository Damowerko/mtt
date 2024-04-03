import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from mtt.models.sparse import SparseBase, SparseData, SparseOutput


class EGNNConv(gnn.MessagePassing):
    def __init__(
        self,
        n_mod: int,
        n_pos: int,
        n_hidden: int,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.m_mod = n_mod
        self.n_mod = n_pos

        self.mod_mlp = gnn.MLP([2 * n_mod + n_pos, n_hidden], act=nn.LeakyReLU())
        self.dir_mlp = gnn.MLP(
            [n_hidden, n_pos * n_pos], plain_last=True, act=nn.LeakyReLU()
        )
        self.update_mlp = gnn.MLP(
            [n_hidden, n_mod], plain_last=True, act=nn.LeakyReLU()
        )

    def forward(
        self,
        f: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        Args:
            x_mod: (N, mod_features_in), node features with no directional or positional information
            x_pos: (N, pos_features_in), node features with positional/directional information
            edge_index: (2, E), edge index
        """
        propagate_out = self.propagate(
            edge_index, f=f, x=x.reshape(-1, x.shape[-2] * 2)
        )
        m, y = torch.split(propagate_out, [self.n_hidden, self.n_pos_out * 2], dim=-1)
        g = self.update_mlp(m)
        y = y.reshape(-1, self.n_pos_out, 2)
        return g, y

    def message(
        self, f_j: torch.Tensor, f_i: torch.Tensor, x_j: torch.Tensor, x_i: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_j: (E, n_mod_in), node features of the src nodes
            f_i: (E, n_mod_in), node features of the dst nodes
            x_j: (E, n_pos_in, 2), positional features of the src
            x_i: (E, n_pos_in, 2), positional features of the dst
        """
        E = f_j.shape[0]
        x_j = x_j.reshape(E, self.n_pos, 2)
        x_i = x_i.reshape(E, self.n_pos, 2)

        f_cat_x = torch.cat([f_j, f_i, (x_i - x_j).norm(dim=-1)], dim=-1)
        # m_ij.shape = (E, n_hidden)
        hidden_ij = self.mod_mlp(f_cat_x)

        x_mat = self.dir_mlp(hidden_ij).reshape(-1, self.n_pos, self.n_pos)
        x_mat = (
            x_mat + torch.eye(self.n_pos, self.n_pos, device=x_mat.device)[None, ...]
        )
        # delta_x.shape is (E, n_pos, 2)
        y_j = x_mat @ (x_i - x_j)
        message = torch.cat([hidden_ij, y_j.reshape(-1, self.n_pos * 2)], dim=-1)
        return message


class EGNN(SparseBase):
    def __init__(
        self,
        n_features=2,
        n_positions=2,
        n_hidden=128,
        n_layers=4,
        ratio=0.5,
        n_neighbors=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_features_in = self.input_length  # time
        self.n_features_out = 2  # sigma, logp
        self.n_positions_in = 2  # measurement_position, sensor_position

        self.n_features = n_features
        self.n_positions = n_positions
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors

        self.f_readin = gnn.MLP(
            [self.n_features_in, n_hidden, n_features], act=nn.LeakyReLU()
        )
        self.f_readout = gnn.MLP(
            [n_features, n_hidden, self.n_features_out],
            act=nn.LeakyReLU(),
            plain_last=True,
        )

        egnn = []
        select = []
        for _ in range(n_layers):
            egnn.append(
                EGNNConv(
                    self.n_features,
                    self.n_positions,
                    n_hidden,
                )
            )
            select.append(gnn.pool.select.SelectTopK(self.n_features, ratio=ratio))

        self.egnn = typing.cast(list[EGNNConv], nn.ModuleList(egnn))
        self.select = typing.cast(
            list[gnn.pool.select.SelectTopK], nn.ModuleList(select)
        )

    def parse_input(self, data: SparseData):
        return data

    def _forward(self, data: SparseData):
        positions = data.measurement_position

        # f_in.shape = (N, self.input_length)
        # 1 hot embedding of time
        f = F.one_hot(data.measurement_time, self.input_length).float()
        f = self.f_readin.forward(f)

        # x_in.shape = (N, 2, 2) the last two dimensions are the x,y coordinates
        x = torch.zeros(data.measurement_position.shape[0], self.n_positions, 2)
        x[:, 0] = data.measurement_position
        x[:, 1] = data.sensor_position

        batch_idx = torch.repeat_interleave(
            torch.arange(
                len(data.measurement_batch_sizes),
                device=data.measurement_batch_sizes.device,
            ),
            data.measurement_batch_sizes,
        )
        for i in range(self.n_layers):
            # compute graph based on the positions
            edge_index = gnn.pool.knn_graph(
                positions, k=self.n_neighbors, batch=batch_idx
            )
            df, dx = self.egnn[i].forward(f, x, edge_index)

            # scale position update
            C = self.n_neighbors
            f = f + df
            x = x + dx / C
            positions = x[:, 0, :]

            # node subsampling
            selection = self.select[i].forward(f, batch_idx)
            x = x[selection.node_index]
            f = f[selection.node_index]
            positions = positions[selection.node_index]
            batch_idx = batch_idx[selection.node_index]

        mu = positions
        sigma = f[:, 0, None].expand(-1, 2)
        logits = f[:, 1]
        batch = torch.bincount(batch_idx)
        return mu, sigma, logits, batch

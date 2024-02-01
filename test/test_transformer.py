import pytest
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data

from mtt.models.transformer import (
    SelectionMechanism,
    SpatialTransformerDecoder,
    SpatialTransformerEncoder,
)


@pytest.fixture
def data():
    x = torch.randn(10, 32)  # Example node features
    pos = torch.randn(10, 2)  # Example node positions
    edge_index = radius_graph(pos, r=0.5, batch=None, loop=True)
    return Data(x=x, pos=pos, edge_index=edge_index)


def test_encoder_forward(data):
    n_channels = 8
    heads = 4
    pos_dim = 2
    n_layers = 2
    dropout = 0.2

    encoder = SpatialTransformerEncoder(n_channels, n_layers, pos_dim, heads, dropout)

    out = encoder.forward(data.x, data.pos, data.edge_index)
    assert out.shape == data.x.shape
    assert out.dtype == torch.float32


def test_selection_mechanism_forward():
    input_dim = 10
    n_channels = 5
    n_hidden = 20
    n_layers = 3
    threshold = 0.5
    dropout = 0.2

    mechanism = SelectionMechanism(
        input_dim, n_channels, n_hidden, n_layers, threshold, dropout
    )

    inputs = torch.randn(32, input_dim)
    embeddings = torch.randn(32, n_channels)
    batch = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    batch_size = 4

    z_updated, object, mask = mechanism.forward(inputs, embeddings, batch, batch_size)
    num_updated = z_updated.shape[0]
    assert z_updated.shape == (num_updated, input_dim)
    assert object.shape == (num_updated, n_channels)


def test_decoder_forward(data):
    torch.manual_seed(0)

    n_channels = 8
    pos_dim = 2
    n_layers = 2
    heads = 4

    N = data.x.shape[0]
    mask = torch.randperm(N) < N // 2
    object = data.x[mask]

    decoder = SpatialTransformerDecoder(n_channels, n_layers, pos_dim, heads)
    output = decoder.forward(data.x, object, data.pos, mask, data.edge_index)
    assert output.shape == data.x[mask].shape

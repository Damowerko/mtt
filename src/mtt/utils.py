from typing import Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def to_cartesian(val) -> NDArray[np.float64]:
    """
    Convert polar coordinates to cartesian coordinates.
    """
    val = np.asarray(val, np.float64)
    shape = val.shape
    val = val.reshape(-1, 2)
    r = val[:, 0]
    theta = val[:, 1]
    assert (np.abs(theta) < 2 * np.pi).all()
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).reshape(shape)


def to_polar(val) -> NDArray[np.float64]:
    """
    Convert cartesion coordinates to polar coordinates.
    Args:
        val: (..., 2) the x and y coordinates.
    """
    val = np.asarray(val, np.float64)
    shape = val.shape
    val = val.reshape(-1, 2)
    r = np.linalg.norm(val, axis=1)
    theta = np.arctan2(val[:, 1], val[:, 0])
    return np.stack([r, theta], axis=1).reshape(shape)


def to_polar_torch(val) -> torch.Tensor:
    """
    Convert cartesion coordinates to polar coordinates.
    """
    val = torch.as_tensor(val, dtype=torch.float64)
    shape = val.shape
    val = val.reshape(-1, 2)
    r = torch.norm(val, dim=1)
    theta = torch.atan2(val[:, 1], val[:, 0])
    return torch.stack([r, theta], dim=1).reshape(shape)


def gaussian(XY, mu, cov):
    """
    Compute the Gaussian density function at some points.
    Args:
        XY: (..., 2) x and y positions of where to sample at.
        mu: (2,) the mean of the Gaussian.
        cov: (2,2) the covariance of the Gaussian.
    Returns:
        (...,) the density at the points.
    """
    XY = np.asarray(XY, np.float64)
    mu = np.asarray(mu, np.float64)
    cov = np.asarray(cov, np.float64)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    gaussian = np.exp(
        -0.5 * ((XY - mu)[..., None, :] @ inv[None, None, ...] @ (XY - mu)[..., None])
    ).squeeze()
    norm = np.sqrt((2 * np.pi) ** 2 * det) * 0.01612901 / (128 / XY.shape[0]) ** 2
    return gaussian / norm


def make_grid(img_size: Union[int, Tuple[int, int]], width):
    """
    Make a grid of points.
    Args:
        img_size: (2,) the size of the image.
        width: the width of the grid.
    Returns:
        XY: (img_size, img_size, 2) the grid of points.
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    X, Y = np.meshgrid(
        np.linspace(-width / 2, width / 2, img_size[0]),
        np.linspace(-width / 2, width / 2, img_size[1]),
    )
    XY = np.stack([X, Y], axis=-1)
    return XY


def compute_ospa_components(
    X: np.ndarray, Y: np.ndarray, cutoff: float, p: int = 2
) -> Tuple[float, float]:
    """
    Compute the OSPA metric.
    Args:
        X: (m, 2) set of positions.
        Y: (n, 2) set of positions.
        cutoff: the cutoff for the OSPA metric.
        p: the p-norm to use.
    Returns:
        ospa_distance: the distance component of the OSPA metric.
        ospa_cardinality: the cardinality component of the OSPA metric.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)

    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[1] == 2
    assert Y.shape[1] == 2

    m = X.shape[0]
    n = Y.shape[0]

    # ospa is symmetric, so we assume m <= n.
    if m > n:
        return compute_ospa_components(Y, X, cutoff)
    if n == 0:
        return 0, 0
    if m == 0:
        return 0, cutoff

    dist = cdist(X, Y, metric="minkowski", p=p) ** p
    xidx, yidx = linear_sum_assignment(dist)
    ospa_distance: float = np.minimum(dist[xidx, yidx], cutoff**p).sum() / n
    # since m <= n, for any unassigned y we have a cost of cutoff
    ospa_cardinality = cutoff**p * (n - m) / n

    # take the p-root since we took the power earlier
    ospa_distance = ospa_distance ** (1 / p)
    ospa_cardinality = ospa_cardinality ** (1 / p)
    return ospa_distance, ospa_cardinality


def compute_ospa(X: np.ndarray, Y: np.ndarray, cutoff: float, p: int = 2) -> float:
    """
    Compute the OSPA metric.
    Args:
        X: (m, 2) set of positions.
        Y: (n, 2) set of positions.
        cutoff: the cutoff for the OSPA metric.
        p: the p-norm to use.
    Returns:
        ospa: the OSPA metric.
    """
    ospa_distance, ospa_cardinality = compute_ospa_components(X, Y, cutoff, p)
    # since this is a take p-root of the sum of powers to p
    return (ospa_distance**p + ospa_cardinality**p) ** (1 / p)

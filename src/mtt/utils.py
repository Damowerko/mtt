import numpy as np
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
    """
    val = np.asarray(val, np.float64)
    shape = val.shape
    val = val.reshape(-1, 2)
    r = np.linalg.norm(val, axis=1)
    theta = np.arctan2(val[:, 1], val[:, 0])
    return np.stack([r, theta], axis=1).reshape(shape)


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
    gaussian = np.exp(-0.5 * np.sum((XY - mu) @ inv * (XY - mu), axis=-1))
    norm = np.sqrt((2 * np.pi) ** 2 * det)
    return gaussian / norm


def ospa(X: np.ndarray, Y: np.ndarray, cutoff: float, p: int = 2) -> float:
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
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)

    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[1] == 2
    assert Y.shape[1] == 2

    m = X.shape[0]
    n = Y.shape[0]

    # ospa is symmetric, so we assume m <= n.
    if n > m:
        return ospa(Y, X, cutoff)
    if n == 0:
        return 0
    if m == 0:
        return cutoff

    dist = cdist(X, Y, metric="minkowski", p=p) ** p
    xidx, yidx = linear_sum_assignment(dist)
    cost: float = np.minimum(dist[xidx, yidx], cutoff ** p).sum()
    # since m <= n, for any unassigned y we have a cost of cutoff
    cost += cutoff ** p * (n - m)
    return (cost / n) ** (1 / p)

import numpy as np


def to_cartesian(val):
    """
    Convert polar coordinates to cartesian coordinates.
    """
    val = np.asarray(val).reshape(-1, 2)
    r = val[:, 0]
    theta = val[:, 1]
    assert (np.abs(theta) < 2 * np.pi).all()
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def to_polar(val):
    """
    Convert cartesion coordinates to polar coordinates.
    """
    val = np.asarray(val).reshape(-1, 2)
    r = np.linalg.norm(val, axis=1)
    theta = np.arctan2(val[:, 1], val[:, 0])
    return np.stack([r, theta], axis=1)

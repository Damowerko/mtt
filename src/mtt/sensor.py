from typing import List

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from mtt.utils import to_cartesian, to_polar, to_polar_torch


class Sensor:
    def __init__(
        self,
        position: ArrayLike = (0, 0),
        noise: ArrayLike = (0.1, 0.1),
        range_max: float = 500.0,
        p_detection: float = 0.9,
    ) -> None:
        """
        Initialize a sensor at a given position with additive noise.
        The sensor measures the range and bearing of targets.

        Args:
            sensor_position: (2,) the position of the sensor.
            noise: (range, bearing) standard deviation of the noise.
        """
        self.position = np.asarray(position).reshape(2)
        self.noise = np.asarray(noise).reshape(2)
        self.p_detection = p_detection
        self.range_max = range_max
        self.rng = np.random.default_rng()

    def measure(self, target_positions: ArrayLike):
        """
        Simulate range and bearing measurements from a sensor at some position with noise.

        Args:
            target_positions: (N, 2) array of the position of the N targets.p

        Returns:
            (N,2) range and bearing measurements.
        """
        target_positions = np.asarray(target_positions, np.float64)
        detected = self.rng.uniform(size=len(target_positions)) < self.p_detection
        detected &= (
            np.linalg.norm(target_positions - self.position[None, :], axis=1)
            <= self.range_max
        )
        target_positions = target_positions[detected]

        measurements = to_polar(target_positions - self.position[None, :])
        measurements += self.rng.normal(0, self.noise, size=measurements.shape)
        return to_cartesian(measurements) + self.position[None, :]

    def measurement_density(
        self, XY, target_measurements, sum=True, jacobian=True
    ) -> NDArray[np.float64]:
        """
        Compute the density function of a measurement as some points.

        Let X be a the RV in the polar cooridnate system and Y = g(X) is the measurement
        in the Cartesian coordinate system. Then the density of Y is given by:
            fy(y) = fx(x) * |J|
        where |J| is the determinant of the Jacobian of g.

        Args:
            XY: (..., 2) x and y positions of where to sample at.
            target_measurements: (..., 2) an ndarray of x,y measured target positions. If `sum` is false then the shape will (shape_XY, shape_target_measurements, 2).

        Returns:
            The value of the density function at the given position.
        """
        rtheta = to_polar(XY - self.position).reshape(-1, 2)
        target_rthetas = to_polar(target_measurements - self.position).reshape(-1, 2)
        delta_r = np.abs(rtheta[:, 0, None] - target_rthetas[:, 0, None].T)
        delta_theta = rtheta[:, 1, None] - target_rthetas[:, 1, None].T
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        Z = np.exp(
            -0.5
            * (
                delta_r**2 / self.noise[0] ** 2
                + delta_theta**2 / self.noise[1] ** 2
            )
        ) / (np.sqrt(2 * np.pi) * self.noise[0] * self.noise[1])
        if sum:
            Z = Z.sum(axis=1)
            out_shape = XY.shape[:-1]
        else:
            out_shape = XY.shape[:-1] + target_measurements.shape[:-1]
        if jacobian:
            Z /= rtheta[:, 0]
        return Z.reshape(out_shape)

    def measurement_density_torch(
        self, XY, target_measurements, device=None
    ) -> torch.Tensor:
        out_shape = XY.shape[:-1]
        XY = XY.reshape(-1, 2)
        target_measurements = target_measurements.reshape(-1, 2)
        sensor_position = torch.as_tensor(self.position, device=device)

        rtheta = to_polar_torch(XY - sensor_position)
        target_rthetas = to_polar_torch(target_measurements - sensor_position)
        delta_r = torch.abs(rtheta[:, 0, None] - target_rthetas[:, 0, None].T)
        delta_theta = rtheta[:, 1, None] - target_rthetas[:, 1, None].T
        delta_theta = (delta_theta + torch.pi) % (2 * torch.pi) - torch.pi
        Z = (
            torch.exp(
                -0.5
                * (
                    delta_r**2 / self.noise[0] ** 2
                    + delta_theta**2 / self.noise[1] ** 2
                )
            ).sum(dim=1)
            / (np.sqrt(2 * torch.pi) * self.noise[0] * self.noise[1])
            / rtheta[:, 0]
        )
        return Z.reshape(out_shape)


def measurement_image(
    size: int,
    window: float,
    sensors: List[Sensor],
    measurements: List[torch.Tensor],
    device=None,
) -> torch.Tensor:
    """
    Image of the density function.

    Args:
        size int: the width and height of the image.
        window float: the size of the window.
        sensors List[Sensor]: the sensors.
        measurements List[torch.Tensor]: a list of measurements for each sensor.
        device torch.device | str | None: the device to use.
    """
    x = torch.linspace(-window / 2, window / 2, size, device=device)
    y = torch.linspace(-window / 2, window / 2, size, device=device)
    XY = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=2)
    Z = torch.zeros((size, size), device=device)
    for s, m in zip(sensors, measurements):
        Z += s.measurement_density_torch(XY, m, device=device)
    return Z.T  # transpose to match image coordinates

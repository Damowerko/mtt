import numpy as np
from numpy.typing import ArrayLike

from mtt.utils import to_cartesian, to_polar

rng = np.random.default_rng()


class Sensor:
    def __init__(
        self,
        position: ArrayLike = (0, 0),
        noise: ArrayLike = (0.1, 0.1),
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

    def measure(self, target_positions: ArrayLike) -> np.ndarray:
        """
        Simulate range and bearing measurements from a sensor at some position with noise.

        Args:
            target_positions: (N, 2) array of the position of the N targets.

        Returns:
            (N,2) range and bearing measurements.
        """
        detected = rng.uniform(size=len(target_positions)) < self.p_detection
        target_positions = target_positions[detected]

        measurements = to_polar(target_positions - self.position[None, :])
        measurements += rng.normal(0, self.noise, size=measurements.shape)
        return to_cartesian(measurements) + self.position[None, :]

    def measurement_density(self, XY, target_measurements) -> np.ndarray:
        """
        Compute the density function of a measurement as some points.

        Let X be a the RV in the polar cooridnate system and Y = g(X) is the measurement
        in the Cartesian coordinate system. Then the density of Y is given by:
            fy(y) = fx(x) * |J|
        where |J| is the determinant of the Jacobian of g.

        Args:
            XY: (..., 2) x and y positions of where to sample at.
            target_measurements: (..., 2) an ndarray of x,y measured target positions.

        Returns:
            The value of the density function at the given position.
        """
        Z = np.zeros(XY.shape[:-1])
        rtheta = to_polar(XY - self.position)
        target_rthetas = to_polar(target_measurements - self.position)
        for target_r, target_theta in target_rthetas.reshape(-1, 2):
            delta_r = np.abs(rtheta[..., 0] - target_r)
            delta_theta = (rtheta[..., 1] - target_theta + np.pi) % (2 * np.pi) - np.pi
            Z += (
                1  # (1 / rtheta[..., 0])
                * np.exp(
                    -0.5
                    * (
                        delta_r ** 2 / self.noise[0] ** 2
                        + delta_theta ** 2 / self.noise[1] ** 2
                    )
                )
                / (np.sqrt(2 * np.pi) * self.noise[0] * self.noise[1])
            )
        return Z

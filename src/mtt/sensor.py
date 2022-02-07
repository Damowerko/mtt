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
            noise: (2,) standard deviation of the noise for the range and bearing.
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
        # range should be positive
        # measurements[:, 0] = np.fmax(0, measurements[:, 0])
        return measurements

    def measurement_density(self, XY, target_measurements) -> np.ndarray:
        """
        Compute the density function of a measurement as some points.

        Let X be a the RV in the polar cooridnate system and Y = g(X) is the measurement
        in the Cartesian coordinate system. Then the density of Y is given by:
            fy(y) = fx(x) * |J|
        where |J| is the determinant of the Jacobian of g.

        Args:
            XY: (...,2) x and y positions of where to sample at.
            target_measurements: (N, 2) an ndarray of x,y measured target positions.

        Returns:
            The value of the density function at the given position.
        """
        Z = np.zeros(XY.shape[:-1])
        XY = XY - self.position[None, None, :]
        r = np.linalg.norm(XY, axis=2)
        theta = np.arctan2(XY[..., 1], XY[..., 0])
        for target_xy in target_measurements:
            delta_r = np.abs(r - target_xy[0])
            delta_theta = (theta - target_xy[1] + np.pi) % (2 * np.pi) - np.pi
            Z += (
                (1 / r)
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

    def measurement_image(self, size: int, target_measurements: np.ndarray):
        """
        Image of the density function.

        Args:
            size int: the width and height of the image.
            target_measurements: (N, 2,) the r,theta (range, bearing) for each target.
        """
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        return self.measurement_density(np.stack((X, Y), axis=2), target_measurements)

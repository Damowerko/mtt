from typing import List, Optional, Tuple, Union
import numpy as np
from mtt.target import Target
from mtt.sensor import Sensor

rng = np.random.default_rng()


class Simulator:
    def __init__(
        self,
        window: float = 1000,
        width: Union[float, None] = None,
        n_targets: float = 10,
        target_lifetime: float = 10,
        clutter_rate: float = 10,
        p_detection: float = 0.95,
        sigma_motion: float = 0.5,
        sigma_initial_state: Tuple[float, float] = (1.0, 1.0),
        n_sensors: float = 5,
        sensor_range: float = 500,
        noise_range: float = 10.0,
        noise_bearing: float = 0.1,
        dt: float = 0.1,
    ):
        """
        Args:
            window: The width and height of the simulation window.
            width: The width of the simulation area.
            n_targets: The average number of targets per km^2.
            n_sensors: The average number of sensors per km^22.
            target_lifetime: The average lifetime of a target in seconds.
            p_clutter: The clutter probability (poisson).
            p_detection: The detection probability.
            sigma_motion: The standard deviation of the target motion.
            sigma_initial_state: The standard deviation of the initial target state.
            sensor_range: The maximum range of the sensors.
            noise_range: The standard deviation of the range measurement noise.
            noise_bearing: The standard deviation of the bearing measurement noise.
            dt: The time step.
        """
        self.window = window
        self.width = window + 2 * sensor_range if width is None else width
        self.area = self.width ** 2 / 1000 ** 2

        self.survival_rate = 1 - 1 / target_lifetime
        self.birth_rate = n_targets / target_lifetime
        self.clutter_rate = clutter_rate
        self.p_detection = p_detection

        self.sigma_motion = sigma_motion
        self.sigma_initial_state = sigma_initial_state
        self.dt = dt

        self.targets = [
            self.init_target()
            for i in range(1 + rng.poisson(n_targets * self.area - 1))
        ]
        self.sensors = [
            self.init_sensor(sensor_range, noise_range, noise_bearing)
            for _ in range(1 + rng.poisson(n_sensors * self.area - 1))
        ]

    def init_target(self) -> Target:
        initial_state = np.concatenate(
            (
                rng.uniform(
                    low=-self.width / 2,
                    high=self.width / 2,
                    size=(2, 1),
                ),
                rng.normal(scale=self.sigma_initial_state, size=(2, 2)),
            ),
            axis=1,
        )
        return Target(initial_state, sigma=self.sigma_motion)

    def init_sensor(
        self, range_max: float, noise_range: float, noise_bearing: float
    ) -> Sensor:
        return Sensor(
            position=rng.uniform(
                low=-self.width / 2,
                high=self.width / 2,
                size=2,
            ),
            noise=(noise_range, noise_bearing),
            p_detection=self.p_detection,
            range_max=range_max,
        )

    @property
    def state(self):
        """
        The state of all the targets as a (N, 2, 3) array where N is the number of targets,
        the second dimension is the x and y components of the state, and the third dimension is
        the position, velocity, and acceleration components of the state.
        """
        if len(self.targets) == 0:
            return np.zeros((0, 2, 3))
        return np.stack([target.state for target in self.targets], axis=0)

    @state.setter
    def state(self, value):
        for target, state in zip(self.targets, value):
            target.state = state

    @property
    def positions(self):
        return self.state[:, :, 0]

    def update(self):
        for target in self.targets:
            target.update(self.dt)

        # Target survival
        survival = rng.uniform(size=len(self.targets)) < self.survival_rate ** self.dt
        self.targets = [
            target for target, alive in zip(self.targets, survival) if alive
        ]

        # Target birth
        n_birth = rng.poisson(self.birth_rate * self.area * self.dt)
        self.targets += [self.init_target() for _ in range(n_birth)]

    def measurements(self):
        measurements = []
        for sensor in self.sensors:
            measurements.append(sensor.measure(self.positions))
        return measurements

    def clutter(self):
        clutter = []
        for _ in self.sensors:
            # p_clutter is the total clutter probability across all sensors
            n_clutter = rng.poisson(self.clutter_rate * self.area / len(self.sensors))
            clutter.append(
                rng.uniform(
                    low=-self.width / 2,
                    high=self.width / 2,
                    size=(n_clutter, 2),
                )
            )
        return clutter

    def position_image(self, size, sigma, target_positions) -> np.ndarray:
        """
        Create an image of the targets at the given positions.

        Args:
            size: The withd and height of the image.
            x: (N,2) The positions of the targets.
        """
        x = target_positions
        X, Y = np.meshgrid(
            np.linspace(-self.window / 2, self.window / 2, size),
            np.linspace(-self.window / 2, self.window / 2, size),
        )
        Z = np.zeros((size, size))
        for i in range(x.shape[0]):
            Z += np.exp(-((X - x[i, 0]) ** 2 + (Y - x[i, 1]) ** 2) * 0.5 / sigma ** 2)
        return Z

    def measurement_image(
        self,
        size: int,
        target_measurements: Optional[List[np.ndarray]] = None,
        clutter: Optional[List[np.ndarray]] = None,
    ):
        """
        Image of the density function.

        Args:
            size int: the width and height of the image.
            target_measurements: list of (N_i, 2) the x,y measurements for each target.
            clutter: list of (N_i, 2) the x,y positions of the clutter.
        """
        if target_measurements is None:
            target_measurements = [np.zeros((0, 2)) for _ in range(len(self.sensors))]
        if clutter is None:
            clutter = [np.zeros((0, 2)) for _ in range(len(self.sensors))]

        x = np.linspace(-self.window / 2, self.window / 2, size)
        y = np.linspace(-self.window / 2, self.window / 2, size)
        XY = np.stack(np.meshgrid(x, y), axis=2)
        Z = np.zeros((size, size))
        for s, m, c in zip(self.sensors, target_measurements, clutter):
            Z += s.measurement_density(XY, np.concatenate((m, c), axis=0))
        return Z

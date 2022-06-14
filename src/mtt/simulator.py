from typing import List, Optional
import numpy as np
from mtt.target import Target
from mtt.sensor import Sensor

rng = np.random.default_rng()


class Simulator:
    def __init__(
        self,
        width=1000,
        p_initial=4,
        p_birth=1e-3,
        p_survival=0.95,
        p_clutter=1e-5,
        p_detection=0.95,
        sigma_motion=0.5,
        sigma_initial_state=(10.0, 1.0, 1.0),
        n_sensors=1,
        noise_range=10.0,
        noise_bearing=0.1,
        dt=0.1,
    ):
        self.p_birth = p_birth
        self.p_survival = p_survival
        self.p_clutter = p_clutter
        self.p_detection = p_detection
        self.sigma_motion = sigma_motion
        self.sigma_initial_state = sigma_initial_state
        self.width = width
        self.noise_range = noise_range
        self.noise_bearing = noise_bearing
        self.dt = dt

        self.targets = [self.init_target() for i in range(rng.poisson(p_initial))]
        # TODO: support multiple sensors
        assert n_sensors == 1, "Only one sensor is supported for now."
        self.sensors = [self.init_sensor()]

    def init_target(self) -> Target:
        initial_state = np.random.normal(0, self.sigma_initial_state, size=(2, 3))
        return Target(initial_state, sigma=self.sigma_motion)

    def init_sensor(self) -> Sensor:
        return Sensor(
            position=(0, 0),
            noise=(self.noise_range, self.noise_bearing),
            p_detection=self.p_detection,
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
        survival = rng.uniform(size=len(self.targets)) < self.p_survival ** self.dt
        self.targets = [
            target for target, alive in zip(self.targets, survival) if alive
        ]

        # Target birth
        n_birth = rng.poisson(self.p_birth * self.dt)
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
            n_clutter = rng.poisson(
                self.p_clutter * self.dt * self.width ** 2 / len(self.sensors)
            )
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
            np.linspace(-self.width / 2, self.width / 2, size),
            np.linspace(-self.width / 2, self.width / 2, size),
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

        x = np.linspace(-self.width / 2, self.width / 2, size)
        y = np.linspace(-self.width / 2, self.width / 2, size)
        XY = np.stack(np.meshgrid(x, y), axis=2)
        Z = np.zeros((size, size))
        for s, m, c in zip(self.sensors, target_measurements, clutter):
            Z += s.measurement_density(XY, np.concatenate((m, c), axis=0))
        return Z

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from mtt.sensor import Sensor
from mtt.target import Target
from mtt.utils import to_cartesian

rng = np.random.default_rng()


class Simulator:
    def __init__(
        self,
        window: float = 1000,
        width: Union[float, None] = None,
        p_birth: float = 0.5,
        p_survival: float = 0.95,
        p_detection: float = 0.95,
        n_sensors: float = 0.25,
        n_clutter: float = 40,
        model="CV",
        sigma_motion: float = 1.0,
        sigma_initial_state: Sequence[float] = (50.0,),
        sensor_range: float = 2000,
        noise_range: float = 10.0,
        noise_bearing: float = 0.035,
        dt: float = 1.0,
    ):
        """
        Args:
            window: The width and height of the simulation window.
            width: The width of the simulation area.
            n_targets: The average number of targets per km^2.
            n_sensors: The average number of sensors per km^22.
            target_lifetime: The average lifetime of a target in seconds.
            clutter_rate: The clutter probability (poisson).
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
        self.area = self.width**2 / 1000**2

        self.p_survival = p_survival
        self.p_birth = p_birth
        self.n_clutter = n_clutter
        self.p_detection = p_detection
        self.sigma_motion = sigma_motion
        self.sigma_initial_state = sigma_initial_state
        self.dt = dt
        self.model = model

        # The lifetime of a target is geometric with p_death=(1-p_survival)
        # this distribution has mean 1/p_death = 1/(1-p_survival)
        n_targets = p_birth / (1 - p_survival)
        # make sure there is at least one target? TODO: check this
        self.targets = [
            self.init_target()
            for i in range(1 + rng.poisson(n_targets * self.area - 1))
        ]
        # make sure there is at least one sensor? TODO: check this
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
                rng.normal(
                    scale=self.sigma_initial_state,
                    size=(2, len(self.sigma_initial_state)),
                ),
            ),
            axis=1,
        )
        return Target(initial_state, sigma=self.sigma_motion, model=self.model)

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
        survival = rng.uniform(size=len(self.targets)) < self.p_survival
        self.targets = [
            target for target, alive in zip(self.targets, survival) if alive
        ]

        # Target birth
        n_birth = rng.poisson(self.p_birth * self.area)
        self.targets += [self.init_target() for _ in range(n_birth)]

    def measurements(self) -> List[np.ndarray]:
        measurements = []
        for sensor in self.sensors:
            measurements.append(sensor.measure(self.positions))
        return measurements

    def clutter(self):
        clutter = []
        for sensor in self.sensors:
            # clutter rate is per one sensor
            n_clutter = rng.poisson(self.n_clutter)

            # measurements in polar coordinates
            rtheta = np.stack(
                (
                    rng.uniform(low=0, high=sensor.range_max, size=n_clutter),
                    rng.uniform(low=0, high=2 * np.pi, size=n_clutter),
                ),
                axis=1,
            )
            XY = to_cartesian(rtheta) - sensor.position
            clutter.append(XY)
        return clutter

    def position_image(
        self, size: int, sigma: float, target_positions: np.ndarray, device=None
    ):
        """
        Create an image of the targets at the given positions.

        Args:
            size: The withd and height of the image.
            sigma: The size of the position blob.
            target_positions: (N,2) The positions of the targets.
            device: The device to move the tensor to.
        """
        x, y = torch.as_tensor(target_positions, device=device).T
        # only consider measurements in windows
        X, Y = torch.meshgrid(
            torch.linspace(-self.window / 2, self.window / 2, size, device=device),
            torch.linspace(-self.window / 2, self.window / 2, size, device=device),
            indexing="ij",
        )
        dx = X.reshape(-1, 1) - x.reshape(1, -1)
        dy = Y.reshape(-1, 1) - y.reshape(1, -1)
        Z = torch.exp(-(dx**2 + dy**2) * 0.5 / sigma**2).sum(dim=1)
        Z /= (2.0 * torch.pi * sigma**2) ** 0.5
        return Z.reshape((size, size)).T  # transpose to match image coordinates

    def measurement_image(
        self,
        size: int,
        target_measurements: Optional[List[np.ndarray]] = None,
        clutter: Optional[List[np.ndarray]] = None,
        device=None,
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
        _target_measurements = [
            torch.from_numpy(m).to(device=device) for m in target_measurements
        ]
        if clutter is None:
            clutter = [np.zeros((0, 2)) for _ in range(len(self.sensors))]
        _clutter = [torch.from_numpy(c).to(device=device) for c in clutter]

        x = torch.linspace(-self.window / 2, self.window / 2, size, device=device)
        y = torch.linspace(-self.window / 2, self.window / 2, size, device=device)
        XY = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=2)
        Z = torch.zeros((size, size), device=device)
        for s, m, c in zip(self.sensors, _target_measurements, _clutter):
            Z += s.measurement_density_torch(
                XY, torch.concat((m, c), dim=0), device=device
            )
        return Z.T  # transpose to match image coordinates

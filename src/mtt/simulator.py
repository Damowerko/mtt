from typing import List
import numpy as np
from mtt.sensor import Sensor
from mtt.target import Target

rng = np.random.default_rng()


class Simulator:
    def __init__(
        self,
        max_targets=10,
        p_initial=4,
        p_birth=1e-3,
        p_survival=0.95,
        sigma_motion=0.5,
        sigma_initial_state=(1.0, 1.0, 1.0),
        max_distance=10,
    ):
        self.max_targets = max_targets
        self.p_birth = p_birth
        self.p_survival = p_survival
        self.sigma_motion = sigma_motion
        self.sigma_initial_state = sigma_initial_state
        self.max_distance = max_distance

        N = rng.poisson(p_initial)
        self.targets = [self.init_target() for i in range(N)]

    def init_target(self) -> Target:
        initial_state = np.random.normal(0, self.sigma_initial_state, size=(2, 3))
        return Target(initial_state, sigma=self.sigma_motion)

    @property
    def state(self):
        """
        The state of all the targets as a (N, 2, 3) array where N is the number of targets,
        the second dimension is the x and y components of the state, and the third dimension is
        the position, velocity, and acceleration components of the state.
        """
        if len(self.targets) == 0:
            return np.zeros((0, 2, 3))
        return np.array([target.state for target in self.targets])

    @state.setter
    def state(self, value):
        for target, state in zip(self.targets, value):
            target.state = state

    @property
    def positions(self):
        return self.state[:, :, 0]

    def update(self, Ts=0.1):
        for target in self.targets:
            target.update(Ts)

        # Target survival
        survival = rng.uniform(size=len(self.targets)) < self.p_survival
        # check within bounds
        survival &= np.linalg.norm(self.positions, axis=1) < self.max_distance
        self.targets = [
            target for target, alive in zip(self.targets, survival) if alive
        ]

        # Target birth
        max_birth = self.max_targets - len(self.targets)
        n_birth = np.fmin(rng.poisson(self.p_birth * Ts), max_birth)
        self.targets += [self.init_target() for _ in range(n_birth)]

    def position_image(self, size, sigma):
        """
        Create an image of the targets at the given positions.

        Args:
            size: The withd and height of the image.
            x: (N,2) The positions of the targets.
        """
        x = self.positions
        X, Y = np.meshgrid(np.linspace(-10, 10, size), np.linspace(-10, 10, size))
        Z = np.zeros((size, size))
        for i in range(x.shape[0]):
            Z += np.exp(-((X - x[i, 0]) ** 2 + (Y - x[i, 1]) ** 2) / sigma)
        return Z

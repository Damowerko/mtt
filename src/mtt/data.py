from collections import deque
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mtt.simulator import Simulator


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        n_steps: int = 100,
        length: int = 20,
        img_size: int = 128,
        sigma_position: float = 10.0,
        init_simulator: Callable[..., Simulator] = Simulator,
        **kwargs,
    ):
        """
        Args:
            n_steps: Number of steps to simulate.
            length: Number of time steps to include in each sample.
            img_size: Size of the image to generate.
            sigma_position: Standard deviation of the Gaussian used to generate the target image.
            init_simulator: Function used to initialize the simulator, should **not** be a lambda.
        """
        super().__init__()
        self.n_steps = n_steps
        self.length = length
        self.img_size = img_size
        self.sigma_position = sigma_position
        self.init_simulator = init_simulator

    def iter_simulation(self, simulator: Optional[Simulator] = None):
        simulator = self.init_simulator() if simulator is None else simulator
        for _ in range(self.n_steps):
            simulator.update()
            # get target positions within the window
            target_positions = simulator.positions[(np.abs(simulator.positions) <= simulator.window / 2).any(axis=1)]
            # sensors can be outside of the window and make detections
            sensor_positions = np.stack([s.position for s in simulator.sensors], axis=0)
            # get measurements and clutter within the window
            measurements =  simulator.measurements()
            clutter = simulator.clutter()
            for i in range(len(measurements)):
                measurements[i] = measurements[i][(np.abs(measurements[i]) <= simulator.window / 2).any(axis=1)]
                clutter[i] = clutter[i][(np.abs(clutter[i]) <= simulator.window / 2).any(axis=1)]
            yield target_positions, sensor_positions, measurements, clutter

    def __iter__(self):
        simulator = self.init_simulator()
        sensor_imgs = deque(maxlen=self.length)
        position_imgs = deque(maxlen=self.length)
        infos = deque(maxlen=self.length)
        for (
            target_positions,
            sensor_positions,
            measurements,
            clutter,
        ) in self.iter_simulation(simulator):
            sensor_imgs.append(
                torch.Tensor(
                    simulator.measurement_image(self.img_size, measurements, clutter)
                )
            )
            position_imgs.append(
                torch.Tensor(
                    simulator.position_image(
                        self.img_size,
                        self.sigma_position,
                        target_positions,
                    )
                )
            )
            infos.append(
                dict(
                    target_positions=target_positions,
                    sensor_positions=sensor_positions,
                    measurements=measurements,
                    clutter=clutter,
                    window=simulator.window,
                )
            )
            if len(sensor_imgs) == self.length:
                yield (
                    torch.stack(tuple(sensor_imgs)),
                    torch.stack(tuple(position_imgs)),
                    list(infos),
                )

    @staticmethod
    def collate_fn(batch):
        return (
            torch.stack(tuple(sensor_imgs for sensor_imgs, _, _ in batch)),
            torch.stack(tuple(positions_imgs for _, positions_imgs, _ in batch)),
            list(infos for _, _, infos in batch),
        )

def _generate_data(online_dataset: OnlineDataset):
    return list(online_dataset.iter_simulation())

def generate_data(online_dataset: OnlineDataset, n_simulations=10):
    futures = []
    with ProcessPoolExecutor() as e:
        for _ in range(n_simulations):
            f = e.submit(_generate_data, online_dataset)
            futures.append(f)
    return [f.result() for f in futures]

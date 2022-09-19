from collections import deque
from typing import Callable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mtt.simulator import Simulator


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        n_steps: int = 10000,
        length: int = 20,
        img_size: int = 256,
        sigma_position=0.01,
        init_simulator: Callable[..., Simulator] = Simulator,
        **kwargs,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.length = length
        self.img_size = img_size
        self.sigma_position = sigma_position
        self.init_simulator = init_simulator

    def __iter__(self):
        simulator = self.init_simulator()

        sensor_imgs = deque(maxlen=self.length)
        position_imgs = deque(maxlen=self.length)
        infos = deque(maxlen=self.length)
        for _ in range(self.n_steps + self.length):
            simulator.update()
            target_positions = simulator.positions
            sensor_positions = np.stack([s.position for s in simulator.sensors], axis=0)
            measurements = simulator.measurements()
            clutter = simulator.clutter()

            sensor_imgs.append(
                torch.Tensor(
                    simulator.measurement_image(self.img_size, measurements, clutter)
                )
            )
            position_imgs.append(
                torch.Tensor(
                    simulator.position_image(
                        self.img_size,
                        self.sigma_position * simulator.width,
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

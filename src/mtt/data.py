import torch
from torch.utils.data import IterableDataset
from collections import deque
from typing import Callable

from mtt.sensor import Sensor
from mtt.simulator import Simulator


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        n_steps: int = 10000,
        length: int = 20,
        img_size: int = 256,
        sigma_position=0.05,
        init_simulator: Callable[..., Simulator] = Simulator(),
        init_sensor: Callable[..., Sensor] = Sensor(),
        image_only=True,
        **kwargs,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.length = length
        self.img_size = img_size
        self.sigma_position = sigma_position
        self.init_simulator = init_simulator
        self.init_sensor = init_sensor
        self.image_only = image_only

    def __iter__(self):
        simulator = self.init_simulator()
        sensor = self.init_sensor()

        inptut_imgs = deque(maxlen=self.length)
        output_imgs = deque(maxlen=self.length)
        target_positions = deque(maxlen=self.length)
        for _ in range(self.n_steps + self.length):
            inptut_imgs.append(
                torch.Tensor(
                    sensor.measurement_image(
                        self.img_size, sensor.measure(simulator.positions)
                    )
                )
            )
            output_imgs.append(
                torch.Tensor(
                    simulator.position_image(
                        self.img_size,
                        self.sigma_position,
                    )
                )
            )
            target_positions.append(simulator.positions)

            if len(inptut_imgs) == self.length:
                out = [
                    torch.stack(tuple(inptut_imgs)),
                    torch.stack(tuple(output_imgs)),
                ]
                if not self.image_only:
                    out.append(tuple(target_positions))
                yield out
            simulator.update()

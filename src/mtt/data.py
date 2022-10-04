from collections import deque
from typing import Callable, List, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mtt.simulator import Simulator


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        n_steps: int = 1000,
        length: int = 20,
        img_size: int = 128,
        sigma_position: float = 10.0,
        init_simulator: Callable[..., Simulator] = Simulator,
        device: str = "cpu",
        dtype=np.float32,
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
        self.device = device
        self.dtype = dtype

    def iter_simulation(self, simulator: Optional[Simulator] = None):
        simulator = self.init_simulator() if simulator is None else simulator
        for _ in range(self.n_steps):
            simulator.update()
            # get target positions within the window
            target_positions = simulator.positions[
                (np.abs(simulator.positions) <= simulator.window / 2).any(axis=1)
            ].astype(self.dtype)
            # sensors can be outside of the window and make detections
            sensor_positions = np.stack(
                [s.position for s in simulator.sensors], axis=0
            ).astype(self.dtype)
            # get measurements and clutter within the window
            measurements = simulator.measurements()
            clutter = simulator.clutter()
            for i in range(len(measurements)):
                measurements[i] = measurements[i][
                    (np.abs(measurements[i]) <= simulator.window / 2).any(axis=1)
                ].astype(self.dtype)
                clutter[i] = clutter[i][
                    (np.abs(clutter[i]) <= simulator.window / 2).any(axis=1)
                ].astype(self.dtype)
            yield target_positions, sensor_positions, measurements, clutter, simulator

    def iter_images(self, simulator: Optional[Simulator] = None):
        simulator = self.init_simulator() if simulator is None else simulator
        with torch.no_grad():
            for (
                target_positions,
                sensor_positions,
                measurements,
                clutter,
                _,
            ) in self.iter_simulation(simulator):
                yield make_image_data(
                    target_positions,
                    sensor_positions,
                    measurements,
                    clutter,
                    simulator,
                    self.img_size,
                    self.sigma_position,
                    self.device,
                )

    def __iter__(self):
        simulator = self.init_simulator()
        sensor_imgs = deque(maxlen=self.length)
        position_imgs = deque(maxlen=self.length)
        infos = deque(maxlen=self.length)
        for (sensor_img, position_img, info) in self.iter_images(simulator):
            sensor_imgs.append(sensor_img)
            position_imgs.append(position_img)
            infos.append(info)
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


def make_image_data(
    target_positions: np.ndarray,
    sensor_positions: np.ndarray,
    measurements: List[np.ndarray],
    clutter: List[np.ndarray],
    simulator: Simulator,
    img_size: int = 128,
    sigma_position: float = 10.0,
    device: str = "cpu",
):
    sensor_img = simulator.measurement_image_torch(
        img_size, measurements, clutter, device=device
    )
    position_img = simulator.position_image_torch(
        img_size,
        sigma_position,
        target_positions,
        device=device,
    )
    info = dict(
        target_positions=target_positions,
        sensor_positions=sensor_positions,
        measurements=measurements,
        clutter=clutter,
        window=simulator.window,
    )
    yield sensor_img, position_img, info


def _generate_data(online_dataset: OnlineDataset, images=False):
    if images:
        return online_dataset.collate_fn(list(online_dataset.iter_images()))
    return list(online_dataset.iter_simulation())


def generate_data(
    online_dataset: OnlineDataset, n_simulations=10, images=False, pool=True
):
    if pool:
        futures = []
        with ProcessPoolExecutor() as e:
            for _ in range(n_simulations):
                f = e.submit(_generate_data, online_dataset, images=images)
                futures.append(f)
        return [f.result() for f in futures]
    else:
        return [
            _generate_data(online_dataset, images=images) for _ in range(n_simulations)
        ]

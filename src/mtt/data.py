import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from glob import glob
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torchdata.datapipes as dp
from torch.utils.data import Dataset, IterableDataset, IterDataPipe

from mtt.simulator import Simulator

SimulationImages = Tuple[torch.Tensor, torch.Tensor, List[Dict]]


def load_simulation_file(path: str) -> SimulationImages:
    return torch.load(path, map_location="cpu")


def simulation_window(data: SimulationImages, length=20) -> List[SimulationImages]:
    """
    Split the simulation data into several overlapping windows of length `length`.

    Args:
        data: the simulation data
        length: the length of the sequences
    """
    sensor_imgs, position_imgs, infos = data
    n_samples = len(sensor_imgs) - length + 1
    samples: List[SimulationImages] = [
        (
            sensor_imgs[i : i + length],
            position_imgs[i : i + length],
            infos[i : i + length],
        )
        for i in range(n_samples)
    ]
    return samples


def collate_fn(batch):
    sensor_imgs, position_imgs, infos = zip(*batch)
    return torch.stack(sensor_imgs), torch.stack(position_imgs), list(infos)


def build_offline_datapipes(
    root_dir="./data/train", length=20
) -> Tuple[IterDataPipe[SimulationImages], IterDataPipe[SimulationImages]]:
    n_files = len(glob(os.path.join(root_dir, "*.pt")))
    datapipes = dp.iter.FileLister(root_dir, "*.pt").random_split(
        weights={"train": 0.95, "val": 0.05}, seed=42, total_length=n_files
    )
    return tuple(
        (
            datapipe.shuffle()
            .sharding_filter()  # we want each shard to have random simulations  # shard based on each file
            # load the simulation data using the above functions
            .map(load_simulation_file)
            .map(partial(simulation_window, length=length))
            # each batch is a collection of random windows from random simulations
            .unbatch()
            .shuffle()
        )
        for datapipe in datapipes
    )


class OfflineDataset(Dataset):
    def __init__(self, length=20, data_dir="./data/simulations/", **kwargs):
        self.length = length
        self.data_dir = data_dir
        # each file contains a single simulation run
        self.n_simulations = len(glob(os.path.join(data_dir, "*.pt")))
        # load an experiment to get the shape of the data
        self.simulation_len = len(self.load_simulation(0)[0]) - length + 1
        print(
            f"Loaded {self.n_simulations} simulations of length {self.simulation_len}."
        )

    def __len__(self) -> int:
        return self.n_simulations * self.simulation_len

    def __getitem__(self, idx: int):
        simulation_idx = idx // self.simulation_len
        simulation = self.load_simulation(simulation_idx)
        start = idx % self.simulation_len
        end = start + self.length
        return (x[start:end] for x in simulation)

    def load_simulation(self, idx) -> Tuple[torch.Tensor, torch.Tensor, List]:
        # load simulation to cpu memory
        return torch.load(os.path.join(self.data_dir, f"{idx}.pt"), map_location="cpu")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


class VectorData(NamedTuple):
    target_positions: npt.NDArray[np.floating]
    sensor_positions: npt.NDArray[np.floating]
    measurements: List[npt.NDArray[np.floating]]
    clutter: List[npt.NDArray[np.floating]]
    simulator: Simulator


class OnlineDataset(IterableDataset):
    def __init__(
        self,
        n_steps: int = 100,
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
                (np.abs(simulator.positions) <= simulator.window / 2).all(axis=1)
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
                    (np.abs(measurements[i]) <= simulator.window / 2).all(axis=1)
                ].astype(self.dtype)
                clutter[i] = clutter[i][
                    (np.abs(clutter[i]) <= simulator.window / 2).all(axis=1)
                ].astype(self.dtype)
            yield VectorData(
                target_positions, sensor_positions, measurements, clutter, simulator
            )

    def vectors_to_images(
        self, target_positions, sensor_positions, measurements, clutter, simulator
    ):
        sensor_img = simulator.measurement_image(
            self.img_size,
            measurements,
            clutter,
            device=self.device,
        )
        position_img = simulator.position_image(
            self.img_size,
            self.sigma_position,
            target_positions,
            device=self.device,
        )
        info = dict(
            target_positions=target_positions,
            sensor_positions=sensor_positions,
            measurements=measurements,
            clutter=clutter,
            window=simulator.window,
        )
        return sensor_img, position_img, info

    def stack_images(self, images, queue=True):
        if queue:
            sensor_imgs = deque(maxlen=self.length)
            position_imgs = deque(maxlen=self.length)
            infos = deque(maxlen=self.length)
            for (sensor_img, position_img, info) in images:
                sensor_imgs.append(sensor_img)
                position_imgs.append(position_img)
                infos.append(info)
                if len(sensor_imgs) == self.length:
                    yield (
                        torch.stack(tuple(sensor_imgs)),
                        torch.stack(tuple(position_imgs)),
                        list(infos),
                    )
        else:
            sensor_imgs, position_imgs, infos = OnlineDataset.collate_fn(list(images))
            for i in range(len(sensor_imgs) - self.length + 1):
                yield (
                    sensor_imgs[i : i + self.length],
                    position_imgs[i : i + self.length],
                    infos[i : i + self.length],
                )

    def iter_images(self, simulator: Optional[Simulator] = None):
        simulator = self.init_simulator() if simulator is None else simulator
        with torch.no_grad():
            for args in self.iter_simulation(simulator):
                yield self.vectors_to_images(*args)

    def __iter__(self):
        return self.stack_images(self.iter_images())

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


def _generate_data(online_dataset: OnlineDataset):
    return list(online_dataset.iter_simulation())


def generate_data(online_dataset: OnlineDataset, n_simulations=10):
    """
    A data generator that can be used to generate data in parallel.
    """
    futures = []
    # we generate the simulations on cpu in parallel
    with ProcessPoolExecutor() as e:
        for _ in range(n_simulations):
            futures += [e.submit(_generate_data, online_dataset)]

    # the images are generated on the gpu in sequence
    # iterate over the futures as they complete
    for f in as_completed(futures):
        vectors = f.result()
        yield online_dataset.collate_fn(
            [online_dataset.vectors_to_images(*v) for v in vectors]
        )

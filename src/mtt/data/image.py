from collections import deque
from functools import partial
from itertools import chain
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterable, List, NamedTuple, Optional, Union

import numpy as np
import torch
import torchdata.datapipes as dp
from torch.utils.data import IterableDataset, IterDataPipe

from mtt.data.sim import SimGenerator, SimulationStep
from mtt.data.utils import rolling_window
from mtt.simulator import Simulator


class ImageData(NamedTuple):
    sensor_images: torch.Tensor
    target_images: torch.Tensor
    info: Dict


class StackedImageData(NamedTuple):
    sensor_images: torch.Tensor
    target_images: torch.Tensor
    info: List[Dict]

    def __len__(self):
        return len(self.sensor_images)

    def __getitem__(self, index):
        return StackedImageData(
            self.sensor_images[index], self.target_images[index], self.info[index]
        )


class StackedImageBatch(NamedTuple):
    sensor_images: torch.Tensor
    target_images: torch.Tensor
    info: List[List[Dict]]


def collate_image_fn(data: Iterable[StackedImageData]) -> StackedImageBatch:
    sensor_imgs, position_imgs, infos = zip(*data)
    return StackedImageData(
        torch.stack(sensor_imgs), torch.stack(position_imgs), list(infos)  # type: ignore
    )


def stack_images(data: Iterable[ImageData]) -> StackedImageData:
    """
    Stack the images from a list of ImageData into a single StackedImageData.
    """
    sensor_imgs, position_imgs, infos = zip(*data)
    return StackedImageData(
        torch.stack(sensor_imgs), torch.stack(position_imgs), list(infos)  # type: ignore
    )


def build_image_dp(
    root_dir="./data/train/images",
    length=20,
    map_location="cpu",
    max_files=None,
) -> IterDataPipe[StackedImageData]:
    root_dir = Path(root_dir)
    filenames = sorted(root_dir.glob("*.pt"))

    # load one file to compute the length
    test_data = load_simulation_file(filenames[0].as_posix(), map_location=map_location)
    test_data = rolling_window(test_data, length=length)
    samples_per_file = len(test_data)  # test_data[0] are the stacks of sensor images
    n_files = min(len(filenames), max_files or len(filenames))

    return (
        dp.map.SequenceWrapper(filenames)
        .shuffle()  # need to shuffle before sharding
        .header(n_files)  # get the first n_files
        .sharding_filter()  # distribute filenames to workers
        .map(partial(load_simulation_file, map_location=map_location))
        .map(partial(rolling_window, length=length))
        .in_batch_shuffle()
        .unbatch()
        .set_length(samples_per_file * n_files)
    )


def to_image(
    data: SimulationStep,
    img_size: int = 128,
    sigma_position: float = 10.0,
    device: str = "cpu",
) -> ImageData:
    sensor_img = data.simulator.measurement_image(
        img_size,
        data.measurements,
        data.clutter,
        device=device,
    )
    position_img = data.simulator.position_image(
        img_size,
        sigma_position,
        data.target_positions,
        device=device,
    )
    info = dict(
        target_positions=data.target_positions,
        sensor_positions=data.sensor_positions,
        measurements=data.measurements,
        clutter=data.clutter,
        window=data.simulator.window_width,
    )
    return ImageData(sensor_img, position_img, info)


def load_simulation_file(
    file: Union[str, BinaryIO], map_location="cpu"
) -> StackedImageData:
    data = torch.load(file, map_location=map_location)
    if isinstance(file, BinaryIO):
        file.close()
    return data


class OnlineImageDataset(IterableDataset):
    def __init__(
        self,
        n_experiments: int = 1,
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
        self.n_experiments = n_experiments
        self.length = length
        self.img_size = img_size
        self.sigma_position = sigma_position
        self.init_simulator = init_simulator
        self.device = device
        self.dtype = dtype

        self.sim_generator = SimGenerator(
            n_steps, init_simulator, window_measurements=True
        )

    def iter_images(self):
        with torch.no_grad():
            for simdata in self.sim_generator:
                yield to_image(simdata)

    def stack_images(self, images: Iterable[ImageData], queue=False):
        if queue:
            sensor_imgs = deque(maxlen=self.length)
            position_imgs = deque(maxlen=self.length)
            infos = deque(maxlen=self.length)
            for sensor_img, position_img, info in images:
                sensor_imgs.append(sensor_img)
                position_imgs.append(position_img)
                infos.append(info)
                if len(sensor_imgs) == self.length:
                    yield StackedImageData(
                        torch.stack(tuple(sensor_imgs)),
                        torch.stack(tuple(position_imgs)),
                        list(infos),
                    )
        else:
            sensor_imgs, position_imgs, infos = stack_images(list(images))
            for i in range(len(sensor_imgs) - self.length + 1):
                yield StackedImageData(
                    sensor_imgs[i : i + self.length],
                    position_imgs[i : i + self.length],
                    infos[i : i + self.length],
                )

    def __iter__(self):
        return chain(
            *(self.stack_images(self.iter_images()) for _ in range(self.n_experiments))
        )

    @staticmethod
    def collate_fn(batch: Iterable[StackedImageData]):
        return collate_image_fn(batch)

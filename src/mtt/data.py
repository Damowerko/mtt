import os
import pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from glob import glob
from typing import (
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import torch
import torchdata.datapipes as dp
from torch.utils.data import IterableDataset, IterDataPipe

from mtt.simulator import Simulator

rng = np.random.default_rng()


class VectorData(NamedTuple):
    target_positions: npt.NDArray[np.floating]
    sensor_positions: npt.NDArray[np.floating]
    measurements: List[npt.NDArray[np.floating]]
    clutter: List[npt.NDArray[np.floating]]
    simulator: Simulator


class ImageData(NamedTuple):
    sensor_images: torch.Tensor
    target_images: torch.Tensor
    info: Dict


class StackedImageData(NamedTuple):
    sensor_images: torch.Tensor
    target_images: torch.Tensor
    info: List[Dict]


def load_simulation_file(
    file: Union[str, BinaryIO], map_location="cpu"
) -> StackedImageData:
    data = torch.load(file, map_location=map_location)
    if isinstance(file, BinaryIO):
        file.close()
    return data


def simulation_window(
    data: StackedImageData, length=20, idx: Union[str, npt.NDArray] = "all"
) -> List[StackedImageData]:
    """
    Split the simulation data into several overlapping windows of length `length`.

    Args:
        data: the simulation data
        length: the length of the sequences
        indices: the indices of the windows to return, if "all" return all windows, if "random" return one random window
    """
    sensor_imgs, position_imgs, infos = data
    n_samples = len(sensor_imgs) - length + 1

    if isinstance(idx, str):
        if idx == "all":
            idx = np.arange(n_samples)
        elif idx == "random":
            idx = rng.integers(n_samples, size=1)
        else:
            raise ValueError(f"Invalid indices: {idx}.")

    samples: List[StackedImageData] = [
        StackedImageData(
            sensor_imgs[i : i + length],
            position_imgs[i : i + length],
            infos[i : i + length],
        )
        for i in idx
    ]
    return samples


def collate_fn(batch: List[ImageData]) -> StackedImageData:
    sensor_imgs, position_imgs, infos = zip(*batch)
    return StackedImageData(
        torch.stack(sensor_imgs), torch.stack(position_imgs), list(infos)  # type: ignore
    )


T = TypeVar("T")


def split_sequence(
    sequence: Sequence[T], weights: Sequence[float]
) -> Sequence[Sequence[T]]:
    """
    Split a sequence into several sequences based on the given splits.

    Args:
        sequence: the sequence to split
        weights: the splits to use, don't have to be normalized
    """
    assert all(w >= 0 for w in weights), "splits must be positive"
    if len(weights) == 0:
        return []
    elif len(weights) == 1:
        return [sequence]
    weights = [0.0] + list(weights)
    # splits_idx start at 0 and end at len(sequence)
    split_idx = np.cumsum(weights)
    split_idx *= len(sequence) / split_idx[-1]
    split_idx = np.round(split_idx).astype(int)
    return [
        sequence[split_idx[i] : split_idx[i + 1]] for i in range(len(split_idx) - 1)
    ]


def build_train_datapipes(
    root_dir="./data/train",
    length=20,
    weights: Sequence[float] = (0.99, 0.01),
    map_location="cpu",
    max_files=None,
) -> Tuple[IterDataPipe[StackedImageData], ...]:
    all_filenames = sorted(glob(os.path.join(root_dir, "*.pt")))
    splits = split_sequence(all_filenames, weights)

    # load one file to compute the length
    test_data = load_simulation_file(all_filenames[0], map_location=map_location)
    test_data = simulation_window(test_data, length=length)
    samples_per_file = len(test_data)  # test_data[0] are the stacks of sensor images
    datapipes = []
    for filenames in splits:
        n_files = min(len(filenames), max_files or len(filenames))
        datapipes.append(
            dp.map.SequenceWrapper(filenames)
            .shuffle()  # need to shuffle before sharding
            .header(n_files)  # get the first n_files
            .sharding_filter()  # distribute filenames to workers
            .map(partial(load_simulation_file, map_location=map_location))
            .map(partial(simulation_window, length=length))
            .in_batch_shuffle()
            .unbatch()
            .set_length(samples_per_file * n_files)
        )
    return tuple(datapipes)


def _transform_simulation(simulation_vectors: List[VectorData], **kwargs):
    simulation_images = [vector_to_image(v, **kwargs) for v in simulation_vectors]
    return collate_fn(simulation_images)


def build_test_datapipe(data_path: str, vector_to_image_kwargs: Dict = {}):
    """
    Build a datapipe that loads in VectorData instead of StackedImageData.
    It will convert the VectorData into ImageData and then into StackedImageData.
    """
    with open(data_path, "rb") as f:
        simulations: List[List[VectorData]] = pickle.load(f)
    return (
        dp.map.SequenceWrapper(simulations)
        .to_iter_datapipe()
        .set_length(len(simulations))
        .sharding_filter()
        .map(partial(_transform_simulation, **vector_to_image_kwargs))
        .map(partial(simulation_window, length=20))
    )


def vector_to_image(
    data: VectorData,
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

    def iter_vectors(self, simulator: Optional[Simulator] = None):
        simulator = self.init_simulator() if simulator is None else simulator
        for _ in range(self.n_steps):
            simulator.update()
            # get target positions within the window
            target_positions = simulator.positions[
                (np.abs(simulator.positions) <= simulator.window_width / 2).all(axis=1)
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
                    (np.abs(measurements[i]) <= simulator.window_width / 2).all(axis=1)
                ].astype(self.dtype)
                clutter[i] = clutter[i][
                    (np.abs(clutter[i]) <= simulator.window_width / 2).all(axis=1)
                ].astype(self.dtype)
            yield VectorData(
                target_positions, sensor_positions, measurements, clutter, simulator
            )

    def vector_to_image(self, data: VectorData) -> ImageData:
        # use the vector_to_image module level method
        return vector_to_image(
            data,
            img_size=self.img_size,
            sigma_position=self.sigma_position,
            device=self.device,
        )

    def stack_images(self, images, queue=False):
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
            for vector_data in self.iter_vectors(simulator):
                yield self.vector_to_image(vector_data)

    def __iter__(self):
        return self.stack_images(self.iter_images())

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


def _generate_vectors(online_dataset: OnlineDataset) -> List[VectorData]:
    return list(online_dataset.iter_vectors())


def generate_vectors(
    online_dataset: OnlineDataset, n_simulations=10
) -> Iterable[List[VectorData]]:
    futures = []
    # we generate the simulations on cpu in parallel
    with ProcessPoolExecutor() as e:
        for _ in range(n_simulations):
            futures += [e.submit(_generate_vectors, online_dataset)]
    for f in as_completed(futures):
        yield f.result()


def generate_images_from_vectors(
    online_dataset: OnlineDataset,
    n_simulations=10,
    vector_generator: Optional[Iterable[List[VectorData]]] = None,
) -> Iterable[StackedImageData]:
    """
    A data generator that can be used to generate data in parallel.
    """
    if vector_generator is None:
        vector_generator = generate_vectors(online_dataset, n_simulations)
    # the images are generated on the gpu in sequence
    # iterate over the futures as they complete
    for vectors in vector_generator:
        images = [online_dataset.vector_to_image(v) for v in vectors]
        yield online_dataset.collate_fn(images)

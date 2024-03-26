import typing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mtt.data.sim import SimulationStep


class SparseData(NamedTuple):
    target_position: torch.Tensor
    target_time: torch.Tensor
    target_batch_sizes: torch.Tensor

    measurement_position: torch.Tensor
    sensor_position: torch.Tensor
    is_clutter: torch.Tensor
    measurement_time: torch.Tensor
    measurement_batch_sizes: torch.Tensor

    @staticmethod
    def from_simulation_step(simdata: SimulationStep, time: int = 0):
        target_position = torch.from_numpy(simdata.target_positions)
        target_time = torch.full((len(target_position),), time)

        measurements = []
        is_clutter = []
        sensor_positions = []
        for i, (m, c) in enumerate(zip(simdata.measurements, simdata.clutter)):
            measurements += [torch.from_numpy(m), torch.from_numpy(c)]
            is_clutter += [
                torch.repeat_interleave(
                    torch.BoolTensor([False, True]), torch.tensor([len(m), len(c)])
                )
            ]
            sensor_positions += [
                torch.from_numpy(simdata.sensor_positions[i]).expand(
                    len(m) + len(c), -1
                )
            ]

        measurement_position = torch.cat(measurements, dim=0)
        sensor_positions = torch.cat(sensor_positions, dim=0)
        measurement_time = torch.full((len(measurement_position),), time)
        is_clutter = torch.cat(is_clutter, dim=0)

        return SparseData(
            target_position,
            target_time,
            torch.tensor([len(target_position)]),
            measurement_position,
            sensor_positions,
            is_clutter,
            measurement_time,
            torch.tensor([len(measurement_position)]),
        )

    @staticmethod
    def cat(data: Iterable["SparseData"], batch=False) -> "SparseData":
        """
        Concatenates a list of SparseData into a single SparseData.

        Args:
            data: list of SparseData to concatenate.
            batch: If True, the batch tensor will be increased by 1 for each element in data.
        """
        # use zip to concatenate each nestedtuple of tensors
        concatenated = SparseData(*(torch.cat(x, dim=0) for x in zip(*data)))
        if batch:
            return concatenated
        else:
            return concatenated._replace(
                target_batch_sizes=concatenated.target_batch_sizes.sum(0, True),
                measurement_batch_sizes=concatenated.measurement_batch_sizes.sum(
                    0, True
                ),
            )


class SparseDataset(Dataset):
    @staticmethod
    def collate_fn(data: Iterable[SparseData]) -> SparseData:
        return SparseData.cat(data, batch=True)

    def __init__(self, data_root: str = "./data/train", length=20, slim=False) -> None:
        """
        Initialize the SparseDataLoader object.

        Args:
            data_root (str): The root directory of the data. Defaults to "./data/train".
            length (int): The length of the data. Defaults to 20.
            slim (bool): Whether to use slim mode. Defaults to False.
                In slim mode, dataset will contain only 1 sample per simulation.
        """
        super().__init__()
        self.length = length
        self.slim = slim

        data_path = Path(data_root)
        self.df_targets = pd.read_parquet(data_path / "targets.parquet")
        self.df_measurements = pd.read_parquet(data_path / "measurements.parquet")
        self.df_sensors = pd.read_parquet(data_path / "sensors.parquet")

        # assume that the number of steps is the same for all simulations
        self.n_steps: int = self.df_targets.index.get_level_values("step_idx").max() + 1
        self.n_simulations: int = (
            self.df_targets.index.get_level_values("sim_idx").max() + 1
        )

    def get(self, sim_idx: int, start_idx: int) -> SparseData:
        end_idx = start_idx + self.length

        idx = pd.IndexSlice[sim_idx, start_idx:end_idx]
        df_targets = self.df_targets.loc[idx, :]
        df_measurements = self.df_measurements.loc[idx, :]
        df_sensors = self.df_sensors.loc[idx, :]

        if len(df_targets) == 0:
            target_positions = torch.zeros((0, 2))
            target_times = torch.zeros((0,))
        else:
            target_positions = torch.from_numpy(
                np.stack(df_targets["target_position"].to_list())
            )
            target_times = torch.from_numpy(
                df_targets.index.get_level_values("step_idx").to_numpy() - start_idx
            )

        if len(df_measurements) == 0:
            measurement_positions = torch.zeros((0, 2))
            is_clutters = torch.zeros((0,), dtype=torch.bool)
            sensor_indices = torch.zeros((0,), dtype=torch.long)
            measurement_times = torch.zeros((0,), dtype=torch.long)
        else:
            measurement_positions = torch.from_numpy(
                np.stack(df_measurements["measurement"].to_list(), axis=0)
            )
            is_clutters = torch.from_numpy(
                np.stack(df_measurements["clutter"].to_list(), axis=0)
            )
            sensor_indices = torch.from_numpy(
                np.stack(df_measurements["sensor_idx"].to_list(), axis=0)
            )
            measurement_times = torch.from_numpy(
                df_measurements.index.get_level_values("step_idx").to_numpy()
                - start_idx
            )

        if len(df_sensors) == 0:
            sensor_positions = torch.zeros((0, 2))
        else:
            sensor_positions = torch.from_numpy(
                np.stack(df_sensors["sensor_position"].to_list(), axis=0)
            )

        return SparseData(
            target_positions.float(),
            target_times,
            torch.tensor([len(target_positions)]),
            measurement_positions.float(),
            sensor_positions[sensor_indices].float(),
            is_clutters,
            measurement_times,
            torch.tensor([len(measurement_positions)]),
        )

    def __len__(self) -> int:
        if self.slim:
            return self.n_simulations
        # only consider full sequences of length self.length
        # eg. if n_steps=20 then we can only return 1 sequence of length 20
        return (self.n_steps - self.length + 1) * self.n_simulations

    def __getitem__(self, index: int) -> SparseData:
        if self.slim:
            sim_idx = index
            start_idx = np.random.randint(0, self.n_steps - self.length)
        else:
            sim_idx = index // (self.n_steps - self.length + 1)
            start_idx = index % (self.n_steps - self.length + 1)
        return self.get(sim_idx, start_idx)


def vector_to_df(
    simdata: List[List[SimulationStep]],
    tqdm_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert VectorData into pandas DataFrames. Willl use a ProcessPoolExecutor to parallelize the conversion.
    """
    with ProcessPoolExecutor() as executor:
        if tqdm_kwargs is not None:
            dfs = zip(
                *tqdm(
                    executor.map(parse_sim, range(len(simdata)), simdata),
                    total=len(simdata),
                    **tqdm_kwargs,
                )
            )
        else:
            dfs = zip(*executor.map(parse_sim, range(len(simdata)), simdata))
        df_targets, df_measurements, df_sensors = tuple(
            typing.cast(pd.DataFrame, pd.concat(df_list)) for df_list in dfs
        )
    return df_targets, df_measurements, df_sensors


def parse_sim(sim_idx: int, steps: list[SimulationStep]):
    df_lists = zip(
        *[parse_step(sim_idx, step_idx, step) for step_idx, step in enumerate(steps)]
    )
    return tuple(pd.concat(df_list) for df_list in df_lists)


def parse_step(sim_idx: int, step_idx: int, step: SimulationStep):
    dfs = step.to_pandas()
    dfs = [
        df.assign(sim_idx=sim_idx, step_idx=step_idx).set_index(["sim_idx", "step_idx"])
        for df in dfs
    ]
    return tuple(dfs)

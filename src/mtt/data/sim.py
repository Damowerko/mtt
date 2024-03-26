from typing import Callable, Iterable, List, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from mtt.simulator import Simulator


class SimulationStep(NamedTuple):
    target_positions: npt.NDArray[np.floating]
    sensor_positions: npt.NDArray[np.floating]
    measurements: List[npt.NDArray[np.floating]]
    clutter: List[npt.NDArray[np.floating]]
    simulator: Simulator

    def to_pandas(self):
        """
        Converts the VectorData into serveral pandas DataFrames.
        Vectors are represented as list[float]

        Returns:
            target_df: a dataframe with columns [target_position]
            measurement_df: a dataframe with columns [sensor_idx, measurement, clutter]
            sensor_df: a dataframe with columns [sensor_idx, sensor_position]
        """
        target_df = pd.DataFrame(
            {
                f"target_position_{i}": self.target_positions[:, i].tolist()
                for i in range(self.target_positions.shape[1])
            }
        )
        measurement_dfs = []
        for i, (m, c) in enumerate(zip(self.measurements, self.clutter)):
            measurement_dfs += [
                # DataFrame with measurements of sensor i
                pd.DataFrame(
                    dict(
                        sensor_idx=i,
                        clutter=False,
                        **{
                            f"measurement_position_{i}": m[:, i]
                            for i in range(m.shape[1])
                        },
                    )
                ),
                # DataFrame with clutter of sensor i
                pd.DataFrame(
                    dict(
                        sensor_idx=i,
                        clutter=True,
                        **{
                            f"measurement_position_{i}": c[:, i]
                            for i in range(c.shape[1])
                        },
                    )
                ),
            ]
        measurement_df = pd.concat(measurement_dfs, ignore_index=True)
        sensor_df = pd.DataFrame(
            dict(
                sensor_idx=range(len(self.sensor_positions)),
                **{
                    f"sensor_position_{i}": self.sensor_positions[:, i].tolist()
                    for i in range(self.sensor_positions.shape[1])
                },
            )
        )
        return target_df, measurement_df, sensor_df


class SimGenerator(Iterable[SimulationStep]):
    def __init__(
        self,
        n_steps: int = 100,
        init_simulator: Callable[..., Simulator] = Simulator,
        window_measurements: bool = True,
    ):
        """
        A class that manages the simulation of multiple steps.

        Args:
            n_steps (int): The number of simulation steps to perform.
            init_simulator (Callable[..., Simulator]): A callable that initializes the simulator.
                Defaults to the `Simulator` class.
            window_measurements (bool): A flag indicating whether to consider only measurements
                within the window. Defaults to True.
        """
        self.init_simulator = init_simulator
        self.n_steps = n_steps
        self.window_measurements = window_measurements

    def __iter__(self, simulator: Simulator | None = None):
        """
        Iterates over the simulation steps.

        Args:
            simulator (Simulator | None): An instance of the simulator. If None, a new
                simulator will be initialized using `init_simulator`.

        Yields:
            SimulationStep: A step in the simulation.
        """
        simulator = simulator or self.init_simulator()
        for _ in range(self.n_steps):
            simulator.update()
            # Implementation of iter_experiments goes here
            target_positions = simulator.positions[
                (np.abs(simulator.positions) <= simulator.window_width / 2).all(axis=1)
            ]
            # sensors can be outside of the window and make detections
            sensor_positions = np.stack([s.position for s in simulator.sensors], axis=0)
            measurements = simulator.measurements()
            clutter = simulator.clutter()
            if self.window_measurements:
                # get measurements and clutter within the window
                for i in range(len(measurements)):
                    measurements[i] = measurements[i][
                        (np.abs(measurements[i]) <= simulator.window_width / 2).all(
                            axis=1
                        )
                    ]
                    clutter[i] = clutter[i][
                        (np.abs(clutter[i]) <= simulator.window_width / 2).all(axis=1)
                    ]
            yield SimulationStep(
                target_positions,
                sensor_positions,
                measurements,
                clutter,
                simulator,
            )

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

T = TypeVar("T")


def rolling_window(
    data: Sequence[T], length=20, idx: Union[str, npt.NDArray] = "all"
) -> List[Sequence[T]]:
    """
    Split a sequence into several overlapping windows of length `length`.

    Args:
        data: the simulation data
        length: the length of the sequences
        indices: the indices of the windows to return, if "all" return all windows, if "random" return one random window
    """
    rng = np.random.default_rng()
    n_samples = len(data) - length + 1

    if isinstance(idx, str):
        if idx == "all":
            idx = np.arange(n_samples)
        elif idx == "random":
            idx = rng.integers(n_samples, size=1)
        else:
            raise ValueError(f"Invalid indices: {idx}.")

    return [data[i : i + length] for i in idx]


def parallel_rollout(
    iterable: Iterable[T],
    n_rollouts=10,
    tqdm_kwargs: dict | None = None,
) -> List[List[T]]:
    """
    Uses a ProcessPoolExecutor to generate n_rollouts in parallel.
    Each rollout is a list of all elements from the iterable.
    Useful for parallelizing simulations.

    Args:
        iterable: the data to be rolled out
        n_rollouts: the number of rollouts to generate
        tqdm_kwargs: optional arguments for tqdm progress bar. None to disable.
    """

    with ProcessPoolExecutor() as e:
        if tqdm_kwargs is not None:
            return list(
                tqdm(
                    e.map(list, [iterable] * n_rollouts),
                    total=n_rollouts,
                    **tqdm_kwargs,
                )
            )
        else:
            return list(e.map(list, [iterable] * n_rollouts))

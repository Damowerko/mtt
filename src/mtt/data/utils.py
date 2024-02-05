from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

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
    iterable: Iterable[T], n_rollouts=10, max_workers=None
) -> Iterable[List[T]]:
    """
    Uses a ProcessPoolExecutor to generate n_rollouts in parallel.
    Each rollout is a list of all elements from the iterable.
    Useful for parallelizing simulations.

    Args:
        iterable: the data to be rolled out
        n_rollouts: the number of rollouts to generate
    """

    futures = []
    with ProcessPoolExecutor(max_workers) as e:
        for _ in range(n_rollouts):
            futures += [e.submit(list, iterable)]
    for f in as_completed(futures):
        yield f.result()

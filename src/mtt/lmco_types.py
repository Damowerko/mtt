from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class SupportsEstimate(Protocol):
    """Protocol for data structures containing :obj:`Filter.extract_states` outputs."""

    estimate: NDArray[np.float_]
    """State estimate array for a single object, having shape `(state_dim, )`."""

    covariance: NDArray[np.float_]
    """State estimate covariance matrix for a single object, having shape `(state_dim, state_dim)`."""


EstimateType = TypeVar("EstimateType", bound=SupportsEstimate)


class CNNFilter(Generic[EstimateType]):

    time: float
    """Time when :py:func:`step` was last called."""

    def step(
        self,
        time: float,
        measurements: Mapping[int, NDArray[np.floating[Any]]],
        sensor_states: Mapping[int, NDArray[np.floating[Any]]],
    ) -> None:
        """Perform a multi-sensor update on this filter's belief state.

        This function should propagate the filter forward in time to :obj:`time` and update its internal belief state using :obj:`measurements`.

        Note:
            If there are no detections in a dwell from a sensor, an array of shape `(0, n_observables)` should still be added to the :obj:`measurements` dict.
            Filters should be designed in this case to apply a missed detection logic when updating that filter's belief state from that sensor.

        Args:
            time (float): Timestamp that :obj:`measurements` were detected at.
                This argument should be monotonically increasing in every call to :py:func:`step` unless the filter is specifically designed to handle out-of-sequence measurements.
                Note that this timestamp can be relative (e.g., 0, 1) or in epoch time (e.g., 1568829863.93).
                Regardless of the type of time stamp, the actual value should be consistent for the life of the filter.
            measurements (Mapping[int, NDArray[np.floating[Any]]]): Multi-object detections received from a single scan.
                The keys of the mapping correspond to the measurement model integer IDs to be fused.
                The values of the mapping are arrays of the measurements detected under each sensor's measurement model.
                Each array provided in this mapping has shape `(num_detections, measurement_dim)`.
            sensor_states (Mapping[int, NDArray[np.floating[Any]]]): Sensor states arrays, keyed by a unique integer IDs.
                Required if the models used to construct this filter need sensor state information at runtime.
                Each array provided must have shape `(state_dim, )`.
                An empty mapping must be provided if sensor state information is not required by the filter's models.
        """
        ...

    def extract_states(self) -> Tuple[float, List[EstimateType]]:
        """Extract all target state estimates at the current time step.

        Returns:
            Tuple[float, Tuple[EstimateType]]: Tuple containing the time that :obj:`extract_states` was called and a tuple of state estimates defined by :obj:`EstimateType`.
            The tuple of state estimates has length of the number of objects that were estimated at that time step.
        """
        ...


def slice_meas_dfs(
    time: float,
    meas_dfs: Mapping[int, pd.DataFrame],
    meas_col_names: Mapping[int, Sequence[str]],
    bin_spacing: float = 0,
    time_col_name: str = "time",
) -> Dict[int, NDArray[np.floating[Any]]]:
    """Extract time-sliced measurement numpy arrays as expected by the filter API from measurement DataFrames.

    For a given time step, this function will take in a dictionary of measurement DataFrames (keyed by the measurement model's unique ID)
    and return a dictionary of measurement numpy arrays required by the :obj:`Filter.step`.

    Warning:
        Time slicing will occur using a simple collation of measurements in the inclusive bounds :obj:`[time - 0.5t_b, time + 0.5*t_b]` where :obj:`t_b` is the :obj:`bin_spacing`.
        These bounds are inclusive to enable :obj:`bin_spacing = 0` to extract all times at exactly time :obj:`time`. If this was an open set, this case would not be possible.
        Although extremely unlikely, time elements exactly at the bounds :obj:`time - 0.5t_b` and :obj:`time + 0.5t_b` could unintentionally be included in more than one calls to this function.

        For example, :obj:`[0.5, 1.0, 1.5, 2.0, 2.5]`. At :obj:`time = 1.0` and :obj:`t_b = 0.5`, the grouping would return measurements at time :obj:`[0.5, 1.0, 1.5]`.
        Then, at :obj:`time = 2.0` and :obj:`t_b = 0.5`, the grouping would return measurements at time :obj:`[1.5, 2.0, 2.5]`.
        Notice that the measurement at time :obj:`1.5` was included twice.
        This edge case should be avoided prior to calling this function by either pre-binning or careful selection of :obj:`bin_spacing`.

    Note:
        Row entries in a a measurement DataFrame containing NaNs are assumed to mean that a scan occurred but no measurements were returned.

    Args:
        time (float): Timestamp at which measurements should be extracted.
        meas_dfs (Mapping[int, pandas.DataFrame]): Mapping of measurement DataFrames keyed by each measurement model's unique ID.
        meas_col_names (Mapping[int, Sequence[str]]): Column names that slice the measurement columns of the DataFrames in :obj:`meas_dfs`, keyed by measurement model unique IDs.
        bin_spacing (float, optional): The spacing interval that will be used to collate state vectors into a time bin.
            Defaults to 0.
        time_col_name (str, optional): String name for the time column in of the DataFrames in :obj:`meas_dfs`.
            Defaults to :obj:`"time"`.

    Returns:
        Dict[int, NDArray[np.floating[Any]]]: Numpy arrays of detections received from a single scan, keyed by the corresponding measurement model UID.
        Each array has shape :obj:`(n_detections, n_observables)`.
    """
    # Extract all measurements from each measurement model at this time
    Z = {}
    t_lbound = time - 0.5 * bin_spacing
    t_ubound = time + 0.5 * bin_spacing
    for uid, meas_df in meas_dfs.items():
        rows = meas_df.loc[meas_df[time_col_name].between(t_lbound, t_ubound)]
        if rows.shape[0] == 0:
            # This measurement model did not scan at this timestep.
            # It has nothing to report.
            continue

        # Populate the measurements into the dictionary for this measurement model at this timestep
        meas_mask = np.isin(meas_df.columns, meas_col_names[uid])
        Z[uid] = rows.loc[~rows.isnull().any(axis=1)].values[:, meas_mask]
    return Z


def slice_state_df(
    time: float,
    state_df: pd.DataFrame,
    state_col_names: Sequence[str],
    bin_spacing: float = 0,
    time_col_name: str = "time",
    uid_col_name: str = "sensor_id",
) -> Dict[int, NDArray[np.floating[Any]]]:
    """Extract a time-sliced sensor state dictionary as expected by the filter API from a sensor state DataFrame.

    For a given time step, this function will take a sensor state DataFrame and return a dictionary of sensor states.

    Note:
        This function assumes that the time slicing returns unique sensor state arrays per sensor (i.e., with the same unique ID).
        If multiple sensor state arrays are returned after time-binning for a given sensor unique ID, the resulting sensor state dictionary will contain the average of those state values.

    Warning:
        Time slicing will occur using a simple collation of measurements in the inclusive bounds :obj:`[time - 0.5t_b, time + 0.5*t_b]` where :obj:`t_b` is the :obj:`bin_spacing`.
        These bounds are inclusive to enable :obj:`bin_spacing = 0` to extract all times at exactly time :obj:`time`. If this was an open set, this case would not be possible.
        Although extremely unlikely, time elements exactly at the bounds :obj:`time - 0.5t_b` and :obj:`time + 0.5t_b` could unintentionally be included in more than one calls to this function.

        For example, :obj:`[0.5, 1.0, 1.5, 2.0, 2.5]`. At :obj:`time = 1.0` and :obj:`t_b = 0.5`, the grouping would return states at time :obj:`[0.5, 1.0, 1.5]`.
        Then, at :obj:`time = 2.0` and :obj:`t_b = 0.5`, the grouping would return states at time :obj:`[1.5, 2.0, 2.5]`.
        Notice that the state at time :obj:`1.5` was included twice.
        This edge case should be avoided prior to calling this function by either pre-binning or careful selection of :obj:`bin_spacing`.

    Args:
        time (float): Timestamp at which measurements should be extracted.
        state_df (pd.DataFrame): DataFrame of sensor state trajectories.
        state_col_names (Sequence[str]): Column names that slice the state columns of :obj:`state_df`.
        bin_spacing (float, optional): The spacing interval that will be used to collate state vectors into a time bin.
            Defaults to 0.
        time_col_name (str, optional): String name for the time column in :obj:`state_df`.
            Defaults to :obj:`"time"`.
        uid_col_name (str, optional): String name for the sensor unique ID column in :obj:`state_df`.
            The values in this column must be provided as an integer, or else they will be cast to integers.
            Defaults to :obj:`"sensor_id"`.

    Returns:
        Dict[int, NDArray[np.floating[Any]]]: Numpy arrays of state vectors, keyed by the corresponding sensor UID.
        Each array has shape :obj:`(state_dim, )`.
    """
    # Extract all state vectors at this time
    t_lbound = time - 0.5 * bin_spacing
    t_ubound = time + 0.5 * bin_spacing
    rows = state_df.loc[state_df[time_col_name].between(t_lbound, t_ubound)]

    # Average the state vectors for each sensor
    avgs = (
        rows.groupby(by=uid_col_name, sort=False).mean().loc[:, list(state_col_names)]
    )

    # Put into dictionary by sensor UID
    states = {}
    for uid, state_series in avgs.iterrows():
        states[int(str(uid))] = state_series.to_numpy()
    return states


def run_filter_batch(
    times: Iterable[float],
    filter_: CNNFilter[EstimateType],
    meas_dfs: Mapping[int, pd.DataFrame],
    meas_cols: Mapping[int, Sequence[str]],
    sensor_states: Optional[pd.DataFrame],
    state_cols: Optional[Sequence[str]],
    bin_spacing: float = 0,
    time_col: str = "time",
    sensor_id_col: str = "sensor_id",
) -> List[List[EstimateType]]:
    """Push multiple measurement data frames through a filter in batch.

    This is an example of how `pytrack` runs a filter, and what functions will be required.

    Args:
        times (Iterable[float]): Timestamps corresponding to each measurement data frame in :obj:`meas_dfs`.
        filter_ (Filter[BeliefType, EstimateType]): Filter instance to apply to the measurements.
        meas_dfs (Mapping[int, pd.DataFrame]): Measurement DataFrames keyed by the measurement model's unique ID.
        meas_cols (Mapping[int, Optional[Sequence[str]]], optional): Column names that slice the measurement columns of the DataFrames in :obj:`meas_dfs`, keyed by measurement model unique IDs.
        sensor_states (Optional[pd.DataFrame]): Sensor state trajectories.
            If not :obj:`None`, DataFrame columns must include :obj:`time_col`, :obj:`"sensor_id"`, and :obj:`state_cols`.
            If :obj:`None`, then empty sensor state dictionaries will be passed into `Filter.step` at each scan interval.
        state_cols (Optional[Sequence[str]]): Column names that slice the state columns in :obj:`sensor_states`.
            If :obj:`None`, should only be used if :obj:`sensor_states` is `None`.
        bin_spacing (float, optional): The spacing interval that will be used to collate measurements into a bin. Defaults to 0.0.
        time_col (str, optional): Name of the time column used in :obj:`sensor_states` and the measurement data frames.
            Defaults to :obj:`"time"`.
        sensor_id_col (str, optional): Name of the sensor unique ID column in :obj:`sensor_states`.
            Defaults to :obj:`"sensor_id"`.

    Returns:
        List[List[EstimateType]]:
            Returns the set of :obj:`Filter.extract_states` outputs at each step time.
    """
    X_est = []

    # Loop over all time values
    for time in times:
        # Get measurements from all sensors for current timestep
        Z = slice_meas_dfs(
            time,
            meas_dfs,
            meas_cols,
            bin_spacing,
            time_col,
        )

        # Extract the sensor states for current timestep
        sensor_states_dict = (
            slice_state_df(
                time,
                sensor_states,
                state_cols,  # type: ignore[arg-type]
                bin_spacing,
                time_col,
                sensor_id_col,
            )
            if sensor_states is not None
            else {}
        )

        # Step the filter with measurements and sensor states
        filter_.step(time, Z, sensor_states_dict)

        # Extract the states from this estimate (we don't need the scan times, because the user knows what they are)
        X_est.append(filter_.extract_states()[1])

    return X_est

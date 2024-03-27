from collections import deque
from typing import Any, List, Mapping, Tuple

import numpy as np
import numpy.typing as npt
import torch
from numpy.typing import NDArray

from mtt.lmco_types import CNNFilter
from mtt.models.convolutional import EncoderDecoder
from mtt.peaks import find_peaks
from mtt.sensor import Sensor, measurement_image
from mtt.utils import to_cartesian


def convert_measurements(
    measurements: npt.NDArray[np.floating], from_lmco: bool = True
):
    """
    Convert measurements from LMCO convention to UPenn convention and back.

    LMCO Convention: (bearing, range) Bearing is clockwise starting at the y-axis.
    UPenn Convention: (range, bearing) Bearing is counter-clockwise starting at the x-axis.

    Args:
        measurements: (N, 2) array of measurements in LMCO/UPenn convention.
    Returns:
        converted_measurements: (N, 2) array of measurements in UPenn/LMCO convention.
    """
    converted_measurements = measurements.copy()
    if from_lmco:
        converted_measurements[:, 0] = measurements[:, 1]
        converted_measurements[:, 1] = np.pi / 2 - measurements[:, 0]
    else:
        converted_measurements[:, 0] = np.pi / 2 - measurements[:, 1]
        converted_measurements[:, 1] = measurements[:, 0]
    return converted_measurements


class CNNEsimate:
    def __init__(
        self, mu: NDArray[np.floating[Any]], cov: NDArray[np.floating[Any]]
    ) -> None:
        self.estimate = mu
        self.covariance = cov


class EncoderDecoderFilter(CNNFilter[CNNEsimate]):
    def __init__(
        self, model: EncoderDecoder, window: float, sensor_kwargs=dict()
    ) -> None:
        """
        Args:
            model (EncoderDecoder): EncoderDecoder model to use for filtering.
            window (float): Size of the window in meters that we are filtering on.
            sensor_kwargs (dict, optional): Keyword arguments to pass to the :py:class:`Sensor` class.
        """
        super().__init__()
        self.model = model
        self.sensor_kwargs = sensor_kwargs
        self.window = window
        self.window_center = (window / 2, window / 2)
        # initialize a queue of input images
        self.queue = deque(
            torch.zeros(model.input_shape, device=model.device, dtype=model.dtype),  # type: ignore
            maxlen=model.input_shape[0],
        )

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
        self.time = time

        # collect a list of sensors and corresponding measurements
        sensors: List[Sensor] = []
        _measurements: List[torch.Tensor] = []
        for sensor_id, sensor_state in sensor_states.items():
            assert len(sensor_state) == 2, "Expected sensor state to be 2D position."
            sensors += [Sensor(position=sensor_state, **self.sensor_kwargs)]
            # convert from LMCO to UPenn convention
            measurements_range_bearing = convert_measurements(
                measurements[sensor_id], from_lmco=True
            )
            measurements_cartesian = (
                to_cartesian(measurements_range_bearing) + sensor_state
            )
            # center the window around 0
            measurements_cartesian -= np.array(self.window_center)
            _measurements += [
                torch.from_numpy(measurements_cartesian)
                .to(self.model.dtype)
                .to(self.model.device)
            ]

        # make an image that represents in measurements for all sensors
        sensor_image = measurement_image(
            self.model.input_shape[1],
            self.window,
            sensors,
            _measurements,
            self.model.device,
        )

        # save the image to queue
        self.queue.append(sensor_image)

    def extract_states(self) -> Tuple[float, List[CNNEsimate]]:
        """Extract all target state estimates at the current time step.

        Returns:
            Tuple[float, Tuple[EstimateType]]: Tuple containing the time that :obj:`extract_states` was called and a tuple of state estimates defined by :obj:`EstimateType`.
            The tuple of state estimates has length of the number of objects that were estimated at that time step.
        """
        # the stack of the past self.model.input_shape[0] images is the input to the model
        input = torch.stack(tuple(self.queue))
        # we only care about the last output of the model, take care of the batch dimension
        with torch.no_grad():
            output: NDArray[np.float32] = (
                self.model.eval()(input[None, ...])[-1, -1, ...].cpu().numpy()
            )
        # find peaks in the output by fitting gaussian mixture model
        gmm = find_peaks(output, self.window, center=self.window_center)
        estimates = [CNNEsimate(mu, cov) for mu, cov in zip(gmm.means, gmm.covariances)]
        return self.time, estimates

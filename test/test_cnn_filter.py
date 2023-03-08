import numpy as np

from mtt.cnn_filter import EncoderDecoderFilter, convert_measurements
from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder
from mtt.utils import to_polar


def test_convert_measurements_inverse():
    measurements = np.array([[0, 1], [np.pi / 2, 1], [np.pi, 1], [3 * np.pi / 2, 1]])
    converted_measurements = convert_measurements(measurements, from_lmco=True)
    assert np.allclose(
        measurements, convert_measurements(converted_measurements, from_lmco=False)
    )


def test_encoder_decoder():
    model = Conv2dCoder.load_from_checkpoint("models/58c6fd8a.ckpt").cuda()
    filter = EncoderDecoderFilter(model, 1000)

    # get dataset to test on
    t = 0
    dataset = OnlineDataset(n_steps=100, device="cpu")
    states = []
    for (
        _,
        sensor_positions,
        measurements,
        clutter,
        simulator,
    ) in dataset.iter_vectors():
        window_center = np.array(
            (simulator.window_width / 2, simulator.window_width / 2)
        )
        # the CNN filter class takes in dicts not arrays
        _measurements = {}
        for i, (m, c) in enumerate(zip(measurements, clutter)):
            _measurements[i] = np.concatenate((m, c), axis=0)
            # convert to relative to sensor position
            _measurements[i] -= sensor_positions[i]
            # convert to polar coordinates
            _measurements[i] = to_polar(_measurements[i])
            # convert to LMCO convention
            _measurements[i] = convert_measurements(_measurements[i], from_lmco=False)

        # offset sensor positions to LMCO convention
        _sensor_positions = {
            i: s for i, s in enumerate(sensor_positions + window_center)
        }

        filter.step(
            t,
            _measurements,
            _sensor_positions,
        )
        t += simulator.dt

        # get filter estimate at this time step
        states += [filter.extract_states()]

    # check that the filter is not outputting nonsense
    mean_cardinality = np.mean([len(s[1]) for s in states])
    assert (
        5 < mean_cardinality < 15
    ), f"Mean cardinality {mean_cardinality} is not within expected range"

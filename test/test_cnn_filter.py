import numpy as np

from mtt.cnn_filter import EncoderDecoderFilter
from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder


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
    ) in dataset.iter_simulation():
        # the CNN filter class takes in dicts not arrays
        _measurements = {
            i: np.concatenate((m, c), axis=0)
            for i, (m, c) in enumerate(zip(measurements, clutter))
        }
        _sensor_positions = {i: s for i, s in enumerate(sensor_positions)}

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

from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from mtt.data import OnlineDataset
from mtt.simulator import Simulator

rng = np.random.default_rng()


def plot_mtt(sensor_imgs, position_imgs, info):
    sensor_img = sensor_imgs[-1]
    position_img = position_imgs[-1]
    target_positions = info[-1]["target_positions"]
    sensor_positions = info[-1]["sensor_positions"]
    measurements = np.concatenate(info[-1]["measurements"])
    clutter = np.concatenate(info[-1]["clutter"])

    width = info[-1]["window"]
    extent = (-width / 2, width / 2, -width / 2, width / 2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(ax)):
        ax[i].plot(*target_positions.T, "r.")
        ax[i].plot(*sensor_positions.T, "b.")
        # ax[i].plot(*measurements.T, "bx")
        ax[i].plot(*clutter.T, "y1")

    ax[0].set_title("Sensor Image")
    ax[0].imshow(sensor_img, extent=extent, origin="lower", cmap="gray_r")

    ax[1].set_title("Groud Truth Image")
    ax[1].imshow(position_img, extent=extent, origin="lower", cmap="gray_r")
    ax[1].legend(
        ["Target", "Sensor", "Detection", "Clutter"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    init_simulator = lambda: Simulator()
    dataset = OnlineDataset(init_simulator=init_simulator, n_steps=100)
    data = list(dataset)

    def draw_frame(data):
        fig = plot_mtt(*data)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return iio.imread(buf)

    with iio.imopen("figures/visualize.mp4", "w") as f:
        with ProcessPoolExecutor() as executor:
            f.write(list(executor.map(draw_frame, data)))

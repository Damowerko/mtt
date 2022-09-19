from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mtt.data import OnlineDataset
from mtt.simulator import Simulator
import imageio.v3 as iio

rng = np.random.default_rng()


def plot_mtt(sensor_imgs, position_imgs, info):
    sensor_img = sensor_imgs[-1]
    position_img = position_imgs[-1]
    target_positions = info[-1]["target_positions"]
    sensor_positions = info[-1]["sensor_positions"]
    measurements = np.concatenate(info[-1]["measurements"])
    clutter = np.concatenate(info[-1]["clutter"])

    width = 1000
    extent = (-width / 2, width / 2, -width / 2, width / 2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(ax)):
        ax[i].plot(*target_positions.T, "r.")
        ax[i].plot(*sensor_positions.T, "b.")
        # ax[i].plot(*measurements.T, "bx")
        # ax[i].plot(*clutter.T, "y1")

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
    init_simulator = lambda: Simulator(
        width=1000,
        n_targets=10,
        target_lifetime=10,
        clutter_rate=10,
        p_detection=0.95,
        sigma_motion=0.5,
        sigma_initial_state=(1.0, 1.0),
        n_sensors=5,
        sensor_range=500,
        noise_range=10.0,
        noise_bearing=0.1,
        dt=0.1,
    )
    dataset = OnlineDataset(
        n_steps=100,
        length=20,
        img_size=256,
        init_simulator=init_simulator,
        sigma_position=0.01,
    )
    data = list(dataset)

    def draw_frame(data):
        fig = plot_mtt(*data)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return iio.imread(buf)

    with iio.imopen("figures/visualize.gif", "w") as f:
        for d in data:
            f.write(draw_frame(d))
        # with ProcessPoolExecutor() as executor:
        #     f.write(list(executor.map(draw_frame, data)))

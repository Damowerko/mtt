from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from mtt.data.image import OnlineImageDataset, StackedImageData
from mtt.simulator import Simulator

rng = np.random.default_rng()


def plot_mtt(
    sensor_img,
    target_img,
    info,
    estimates=None,
    plot_measurements=False,
    plot_clutter=False,
):
    target_positions = info["target_positions"]
    sensor_positions = info["sensor_positions"]
    measurements = np.concatenate(info["measurements"])
    clutter = np.concatenate(info["clutter"])
    width = info["window"]
    extent = (-width / 2, width / 2, -width / 2, width / 2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(ax)):
        # on both plots
        ax[i].plot(*target_positions.T, "r.", label="Target")
        # on the first plot
        if i == 0:
            ax[i].plot(*sensor_positions.T, "go", label="Sensor")
            if plot_measurements:
                ax[i].plot(*measurements.T, "bx", label="Sensor Measurement")
            if plot_clutter:
                ax[i].plot(*clutter.T, "y1", label="Clutter")
        # on the second plot
        if i == 1:
            if estimates is not None:
                ax[i].plot(*estimates.T, "b.", label="Estimate")

    ax[0].set_title("Sensor Image")
    ax[0].imshow(sensor_img, extent=extent, origin="lower", cmap="gray_r")

    ax[1].set_title("Output Image")
    ax[1].imshow(target_img, extent=extent, origin="lower", cmap="gray_r")
    # ax[1].legend(
    #     loc="center left",
    #     bbox_to_anchor=(1, 0.5),
    # )
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    init_simulator = lambda: Simulator()
    dataset = OnlineImageDataset(init_simulator=init_simulator, n_steps=100)
    data = list(dataset)

    def draw_frame(data: StackedImageData):
        fig = plot_mtt(data.sensor_images[-1], data.target_images[-1], data.info[-1])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return iio.imread(buf)

    with iio.imopen("figures/visualize.mp4", "w") as f:
        with ProcessPoolExecutor() as executor:
            f.write(list(executor.map(draw_frame, data)))

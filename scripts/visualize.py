import numpy as np
import matplotlib.pyplot as plt
from mtt.sensor import Sensor
from mtt.simulator import Simulator
from mtt.utils import to_cartesian
from mtt.data import OnlineDataset

rng = np.random.default_rng()


if __name__ == "__main__":
    init_simulator = lambda: Simulator(
            width=1000,
            p_initial=4,
            p_birth=1,
            p_survival=0.95,
            p_clutter=1e-4,
            p_detection=0.95,
            sigma_motion=0.5,
            sigma_initial_state=(50.0, 40.0, 2.0),
            n_sensors=1,
            noise_range=10.0,
            noise_bearing=0.1,
            dt=0.1,
    )
    dataset = OnlineDataset(
        n_steps=100,
        length=20,
        img_size=256,
        init_simulator=init_simulator,
    )

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    running = True
    pause = False

    def key_press(event):
        """
        On press event handler.
        """
        if event.key == "q":
            global running
            running = False
        elif event.key == " ":
            global pause
            pause = not pause

    fig.canvas.mpl_connect("key_press_event", key_press)
    for sensor_imgs, position_imgs, info in dataset:
        sensor_img = sensor_imgs[-1]
        position_img = position_imgs[-1]
        target_positions = info[-1]["target_positions"]
        measurements = np.concatenate(info[-1]["measurements"])
        clutter = np.concatenate(info[-1]["clutter"])

        width = 1000
        extent = (-width / 2, width / 2, -width / 2, width / 2)

        if not running:
            break
        # Animate the plot of target velocities over time.
        if pause:
            plt.pause(0.1)
            continue
        for i in range(len(ax)):
            ax[i].clear()
            ax[i].plot(*target_positions.T, "r.")
            ax[i].plot(*measurements.T, "bx")
            ax[i].plot(*clutter.T, "w1")
        ax[0].imshow(sensor_img, extent=extent, origin="lower")
        ax[1].imshow(position_img, extent=extent, origin="lower")
        ax[1].legend(["Target", "Sensor", "Clutter"])
        plt.pause(0.1)

import numpy as np
import matplotlib.pyplot as plt
from mtt.sensor import Sensor
from mtt.simulator import Simulator
from mtt.utils import to_cartesian
from mtt.data import OnlineDataset

rng = np.random.default_rng()


if __name__ == "__main__":
    init_simulator = lambda: Simulator(
        max_targets=10,
        p_initial=4,
        p_birth=2,
        p_survival=0.95,
        sigma_motion=0.1,
        sigma_initial_state=(3.0, 1.0, 1.0),
        max_distance=1e6,
    )
    init_sensor = lambda: Sensor(position=(0, 0), noise=(0.2, 0.1), p_detection=0.9)
    dataset = OnlineDataset(
        n_steps=100,
        length=20,
        img_size=256,
        init_simulator=init_simulator,
        init_sensor=init_sensor,
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
    for sensor_img, target_img, target_position in dataset:
        if not running:
            break
        # Animate the plot of target velocities over time.
        if pause:
            plt.pause(0.1)
            continue
        for i in range(len(ax)):
            ax[i].clear()
            ax[i].plot(target_position[0][:, 0], target_position[0][:, 1], "r.")
        ax[0].imshow(sensor_img[0], extent=(-10, 10, -10, 10), origin="lower")
        ax[1].imshow(target_img[0], extent=(-10, 10, -10, 10), origin="lower")
        ax[1].legend(["Target", "Sensor"])
        plt.pause(0.1)

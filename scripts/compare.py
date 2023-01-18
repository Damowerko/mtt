import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm, trange

from mtt.data import OnlineDataset, VectorData
from mtt.gmphd import gmphd_filter, positions_from_gmphd
from mtt.models import Conv2dCoder, load_model
from mtt.peaks import find_peaks
from mtt.simulator import Simulator
from mtt.smc_phd import SMCPHD
from mtt.utils import compute_ospa

rng = np.random.default_rng()

n_trials = 100  # number of simulations to run
scale = 1  # the width of the area in km
n_peaks_scale = 2.3  # older models' output should be scaled by 2.3
queue = False  # should a deque be used to stack images
gmphd = scale == 1  # run the gmphd filter as well?
smcphd = scale == 1  # run the smcphd filter as well?

data_dir = f"data/test/{scale}km"
simulations_file = os.path.join(data_dir, "simulations.pkl")
gmphd_file = os.path.join(data_dir, "gmphd.pkl")
smcphd_file = os.path.join(data_dir, "smcphd.pkl")


def init_simulator():
    return Simulator(window=1000 * scale)


online_dataset = OnlineDataset(
    n_steps=119,
    sigma_position=10,
    length=20,
    img_size=128 * scale,
    device="cuda",
    init_simulator=init_simulator,
)
simulator = online_dataset.init_simulator()
expected_detections_per_target = (
    simulator.n_sensors * np.pi * simulator.sensors[0].range_max ** 2 / 1000**2
)
# the CNN output is undefined for the first online_dataset.length + 1 time steps
prediction_length = online_dataset.n_steps - online_dataset.length + 1

# run or load simulations
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def run_simulation(*_):
    return list(online_dataset.iter_simulation())


if not os.path.exists(simulations_file):
    with ProcessPoolExecutor() as executor:
        simulations = list(
            tqdm(
                executor.map(run_simulation, range(n_trials)),
                total=n_trials,
                desc="Running simulations",
            )
        )
    with open(simulations_file, "wb") as f:
        pkl.dump(simulations, f)
else:
    with open(simulations_file, "rb") as f:
        simulations: List[List[VectorData]] = pkl.load(f)[:n_trials]
        for i in range(len(simulations)):
            for j in range(len(simulations[i])):
                simulations[i][j] = VectorData(*simulations[i][j])

# for each filter store predicted positions in a dictionary
predicted_positions: Dict[str, List[List[npt.NDArray[np.float64]]]] = {}


def run_gmphd(simulation: List[VectorData]):
    # PHD filter testing
    mixtures = gmphd_filter(simulation)
    # extract positions from gaussian mixtures
    return [
        positions_from_gmphd(mixture, expected_detections_per_target)
        for mixture in mixtures
    ][-prediction_length:]


if gmphd:
    if not os.path.exists(gmphd_file):
        # run phd filter on each simulation
        with ProcessPoolExecutor() as executor:
            predicted_positions["GM-PHD"] = list(
                tqdm(
                    executor.map(run_gmphd, simulations),
                    total=len(simulations),
                    desc="Running PHD filter",
                )
            )
        with open(gmphd_file, "wb") as f:
            pkl.dump(predicted_positions["GM-PHD"], f)
    else:
        with open(gmphd_file, "rb") as f:
            predicted_positions["GM-PHD"] = pkl.load(f)[:n_trials]


def run_smcphd(simulation: List[VectorData], *_, **__):
    simulator = simulation[0].simulator
    smc_phd = SMCPHD(simulator)

    positions: List[npt.NDArray[np.floating]] = []
    for d in simulation:
        smc_phd.step(d.measurements)
        # now, extract the position only
        positions.append(smc_phd.extract_states()[:, :2])
    return positions[-prediction_length:]


if smcphd:
    if not os.path.exists(smcphd_file):
        # run phd filter on each simulation
        with ProcessPoolExecutor() as executor:
            predicted_positions["SMC-PHD"] = list(
                tqdm(
                    executor.map(run_smcphd, simulations),
                    total=len(simulations),
                    desc="Running SMC-PHD filter",
                )
            )
        with open(smcphd_file, "wb") as f:
            pkl.dump(predicted_positions["SMC-PHD"], f)
    else:
        with open(smcphd_file, "rb") as f:
            predicted_positions["SMC-PHD"] = pkl.load(f)[:n_trials]

# Now run the CNN model.
model = load_model(Conv2dCoder, "58c6fd8a")


def run_cnn(simulation: List[VectorData]):
    predictions_cnn: List[npt.NDArray] = []
    with torch.no_grad():
        images = map(online_dataset.vectors_to_images, *zip(*simulation))
        window = simulation[0].simulator.window
        filt_idx = -1
        for sensor_imgs, _, _ in online_dataset.stack_images(images, queue=queue):
            pred_imgs = model(sensor_imgs.cuda()).cpu().numpy()
            # fit a gmm and get the mean of each gaussian in the gmm
            predictions_cnn += [
                find_peaks(
                    pred_imgs[filt_idx], width=window, n_peaks_scale=n_peaks_scale
                )[0]
            ]
    return predictions_cnn


# run the CNN, since we use GPU we do not parallelize
predicted_positions["CNN"] = list(
    tqdm(
        map(run_cnn, simulations),
        total=len(simulations),
        desc="Running CNN filter",
    )
)

# extract the true positions
true_positions: List[List[npt.NDArray[np.floating]]] = []
for simulation in simulations:
    true_positions += [
        [data.target_positions for data in simulation[-prediction_length:]]
    ]

ospa = {k: np.zeros((n_trials, prediction_length)) for k in predicted_positions}
# find the OSPA distance between the predicted positions and the target positions
for simulation_idx in trange(n_trials, desc="Calculating OSPA.", unit="simulation"):
    for k in predicted_positions:
        for step_idx in range(prediction_length):
            x = predicted_positions[k][simulation_idx][step_idx]
            y = true_positions[simulation_idx][step_idx]
            ospa[k][simulation_idx, step_idx] = compute_ospa(x, y, 500)

for k in ospa:
    print(f"{k} OSPA: {np.mean(ospa[k]):.4f} ± {np.std(ospa[k]):.4f}")

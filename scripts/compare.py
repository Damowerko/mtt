import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import ttest_rel
from tqdm import tqdm, trange

from mtt.data import OnlineDataset, VectorData
from mtt.gmphd import gmphd_filter, positions_from_gmphd
from mtt.models import Conv2dCoder, load_model
from mtt.peaks import find_peaks
from mtt.simulator import Simulator
from mtt.smc_phd import SMCPHD
from mtt.utils import compute_ospa_components

rng = np.random.default_rng()

n_trials = 5  # number of simulations to run
scale = 1  # the width of the area in km
n_peaks_scale = 2.3  # older models' output should be scaled by 2.3
queue = False  # should a deque be used to stack images

# which if any phd filters should be run
phd_enable = scale == 1

data_dir = f"data/test/{scale}km"
simulations_file = os.path.join(data_dir, "simulations.pkl")


def init_simulator():
    return Simulator(window_width=1000 * scale)


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


def run_simulation(*_):
    return list(online_dataset.iter_simulation())


# SIMULATIONS
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
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


def run_gmphd(simulation: List[VectorData]):
    # PHD filter testing
    mixtures = gmphd_filter(simulation)
    # extract positions from gaussian mixtures
    return [
        positions_from_gmphd(mixture, expected_detections_per_target)
        for mixture in mixtures
    ][-prediction_length:]


def run_smcphd(simulation: List[VectorData], adaptive_birth: bool = False):
    if not adaptive_birth:
        raise NotImplementedError(
            "SMC-PHD without adaptive birth is no longer implemented."
        )
    simulator = simulation[0].simulator
    smc_phd = SMCPHD(simulator, particles_per_target=1000, particles_per_measurement=10)

    positions: List[npt.NDArray[np.floating]] = []
    for d in simulation:
        smc_phd.step(d.measurements)
        # now, extract the position only
        positions.append(smc_phd.extract_states()[:, :2])
    return positions[-prediction_length:]


run_filter_funcitons = {
    "GM-PHD": run_gmphd,
    "SMC-PHD": partial(run_smcphd, adaptive_birth=False),
    "SMC-PHD (adaptive)": partial(run_smcphd, adaptive_birth=True),
}
# for each filter store predicted positions in a dictionary
predicted_positions: Dict[str, List[List[npt.NDArray[np.float64]]]] = {}
for name, run_phd in run_filter_funcitons.items():
    if not phd_enable:
        continue
    filename = os.path.join(data_dir, f"{name}.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            predicted_positions[name] = pkl.load(f)[:n_trials]
        continue

    # run phd filter on each simulation
    with ProcessPoolExecutor() as executor:
        predicted_positions[name] = list(
            tqdm(
                executor.map(run_filter_funcitons[name], simulations),
                total=len(simulations),
                desc=f"Running {name} filter",
            )
        )
    with open(filename, "wb") as f:
        pkl.dump(predicted_positions[name], f)

# Now run the CNN model.
model = load_model(Conv2dCoder, "58c6fd8a")


def run_cnn(simulation: List[VectorData]):
    predictions_cnn: List[npt.NDArray] = []
    with torch.no_grad():
        images = map(online_dataset.vectors_to_images, *zip(*simulation))
        window = simulation[0].simulator.window_width
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


cnn_filename = os.path.join(data_dir, "CNN.pkl")
if os.path.exists(cnn_filename):
    with open(cnn_filename, "rb") as f:
        predicted_positions["CNN"] = pkl.load(f)[:n_trials]
else:
    # run the CNN, since we use GPU we do not parallelize
    predicted_positions["CNN"] = list(
        tqdm(
            map(run_cnn, simulations),
            total=len(simulations),
            desc="Running CNN filter",
        )
    )
    with open(cnn_filename, "wb") as f:
        pkl.dump(predicted_positions["CNN"], f)

# extract the true positions
true_positions: List[List[npt.NDArray[np.floating]]] = []
for simulation in simulations:
    true_positions += [
        [data.target_positions for data in simulation[-prediction_length:]]
    ]


ospa_components = {
    k: np.zeros((n_trials, prediction_length, 2)) for k in predicted_positions
}
cardinality_error = {
    k: np.zeros((n_trials, prediction_length)) for k in predicted_positions
}
# find the OSPA distance between the predicted positions and the target positions
for simulation_idx in trange(n_trials, desc="Calculating OSPA.", unit="simulation"):
    for k in predicted_positions:
        for step_idx in range(prediction_length):
            x = predicted_positions[k][simulation_idx][step_idx]
            y = true_positions[simulation_idx][step_idx]
            ospa_components[k][simulation_idx, step_idx] = compute_ospa_components(
                x, y, 500, p=2
            )
            cardinality_error[k][simulation_idx, step_idx] = np.abs(len(x) - len(y))

ospa = {}
for k in ospa_components:
    ospa_distance, ospa_cardinality = ospa_components[k].mean(axis=(0, 1))
    # we take the mean of the OSPA distance for each experiment
    ospa[k] = ((ospa_components[k] ** 2).sum(axis=2) ** 0.5).mean(axis=1)

    print(f"\n --- {k} filter results ---")
    print(f"OSPA: {np.mean(ospa[k]):.4f} ± {np.std(ospa[k]):.4f}")
    print(f"OSPA Distance: {ospa_distance:.4f}")
    print(f"OSPA Cardinality: {ospa_cardinality:.4f}")
    print(f"Cardinality Error: {np.mean(cardinality_error[k]):.4f}")

# paired t-test for CNN mean being lower than the other filters
print(f"\n --- Paired t-test for CNN mean being lower than the other filters ---")
if "CNN" in ospa:
    for k in ospa:
        if k == "CNN":
            continue
        test_result = ttest_rel(ospa["CNN"], ospa[k], alternative="less")
        print(f"pvalue for CNN < {k}: {test_result.pvalue}")  # type: ignore
        ospa_mean = ospa[k].mean()
        improvement = -100 * (test_result.confidence_interval(0.99).high) / ospa_mean
        print(f"% Improvement with 99% confidence: {improvement:0.2f}")

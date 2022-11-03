import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from stonesoup.types.state import TaggedWeightedGaussianState

from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder, load_model
from mtt.peaks import find_peaks
from mtt.simulator import Simulator
from mtt.utils import ospa
from mtt.phd import phd_filter, positions_from_phd

rng = np.random.default_rng()

n_trials = 10  # number of simulations to run
scale = 10  # the width of the area in km
queue = False  # should a deque be used to stack images
phd = scale == 1  # run the phd filter as well?

data_dir = f"data/test/{scale}km"
simulations_file = os.path.join(data_dir, "simulations.pkl")
phd_file = os.path.join(data_dir, "phd.pkl")


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
n_detections = (
    simulator.n_sensors * np.pi * simulator.sensors[0].range_max ** 2 / 1000**2
)


def run_simulation(*_):
    return list(online_dataset.iter_simulation())


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
        dataset_vectors = pkl.load(f)[:n_trials]
if phd and not os.path.exists(phd_file):
    if not os.path.exists(phd_file):
        # run phd filter on each simulation
        with ProcessPoolExecutor() as executor:
            predictions_phd = list(
                tqdm(
                    executor.map(phd_filter, simulations),
                    total=len(simulations),
                    desc="Running PHD filter",
                )
            )
        with open(phd_file, "wb") as f:
            pkl.dump(predictions_phd, f)
    else:
        with open(phd_file, "rb") as f:
            predictions_phd = pkl.load(f)


# PHD filter testing
def predict_phd(phds: List[List[TaggedWeightedGaussianState]]):
    return [positions_from_phd(phd, n_detections) for phd in phds]


if phd:
    with open(os.path.join(data_dir, "phd.pkl"), "rb") as f:
        dataset_phd = pkl.load(f)[:n_trials]
    ospa_phd = []
    for dataset_idx in tqdm(range(len(dataset_phd)), desc="Computing PHD stats."):
        vectors = dataset_vectors[dataset_idx][online_dataset.length - 1 :]
        phds = dataset_phd[dataset_idx][online_dataset.length - 1 :]
        predictions_phd = predict_phd(phds)
        ospa_phd.append([])
        for idx in range(len(vectors)):
            target_positions = vectors[idx][0]
            ospa_phd[-1] += [ospa(target_positions, predictions_phd[idx], 500)]
    print("PHD Mean/Variance: ", np.mean(ospa_phd), np.std(ospa_phd))

# CNN model testing
model = load_model(Conv2dCoder, "58c6fd8a")


def predict_cnn(vectors):
    with torch.no_grad():
        images = map(online_dataset.vectors_to_images, *zip(*vectors))
        simulator = dataset_vectors[0][0][4]
        window = simulator.window
        filt_idx = -1
        predictions_cnn = []
        for sensor_imgs, _, _ in online_dataset.stack_images(images, queue=queue):
            pred_imgs = model(sensor_imgs.cuda()).cpu().numpy()
            predictions_cnn += [find_peaks(pred_imgs[filt_idx], width=window)[0]]
    return predictions_cnn


ospa_cnn = []
for dataset_idx in tqdm(range(len(dataset_vectors)), desc="Running CNN"):
    vectors = dataset_vectors[dataset_idx][online_dataset.length - 1 :]
    predictions_cnn = predict_cnn(dataset_vectors[dataset_idx])
    ospa_cnn.append([])
    for idx in range(len(predictions_cnn)):
        target_positions = vectors[idx][0]
        ospa_cnn[-1] += [ospa(target_positions, predictions_cnn[idx], 500)]
print("CNN Mean/Variance: ", np.mean(ospa_cnn), np.std(ospa_cnn))

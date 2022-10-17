import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder
from mtt.peaks import find_peaks
from mtt.simulator import Simulator
from mtt.utils import ospa
from mtt.phd import phd_filter

rng = np.random.default_rng()

n_trials = 100
scale = 2
phd = scale == 1

data_dir = f"data/test/{scale}km"
simulations_file = os.path.join(data_dir, "simulations.pkl")
phd_file = os.path.join(data_dir, "phd.pkl")


def init_simulator():
    return Simulator(window=1000 * scale)


if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(simulations_file):
    simulations = []
    for i in range(n_trials):
        online_dataset = OnlineDataset(
            n_steps=119, length=20, img_size=128 * 2, device="cuda"
        )
        simulations.append(list(online_dataset.iter_simulation()))
    with open(simulations_file, "wb") as f:
        pkl.dump(simulations, f)

    if phd:
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


# model_id = "58c6fd8a"
# run_path = !guild ls {model_id}
# run_path = run_path[0][:-1]
# checkpoint_path = os.path.join(run_path, "checkpoints/best.ckpt")
# model = Conv2dCoder.load_from_checkpoint(checkpoint_path).cuda()

# online_dataset = OnlineDataset(n_steps=119, sigma_position=10, length=20, img_size=128, device="cuda")
# with open("../data/test/simulations.pkl", "rb") as f, open("../data/test/phd.pkl", "rb") as g:
#     dataset_vectors = pkl.load(f)
#     dataset_phd = pkl.load(g)

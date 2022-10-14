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
from mtt.utils import ospa
from mtt.phd import phd_filter

rng = np.random.default_rng()

n_trials = 100
data_dir = "data/test/"
simulations_file = os.path.join(data_dir, "simulations.pkl")
phd_file = os.path.join(data_dir, "phd.pkl")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(simulations_file):
    simulations = []
    for i in range(n_trials):
        online_dataset = OnlineDataset(
            n_steps=119, length=20, img_size=128, device="cuda"
        )
        simulations.append(list(online_dataset.iter_simulation()))

    # run phd filter on each simulation
    with ProcessPoolExecutor() as executor:
        predictions_phd = list(
            tqdm(
                executor.map(phd_filter, simulations),
                total=len(simulations),
                desc="Running PHD filter",
            )
        )

    with open(simulations_file, "wb") as f, open(phd_file, "wb") as g:
        pkl.dump(simulations, f)
        pkl.dump(predictions_phd, g)

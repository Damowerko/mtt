import os
import argparse

import tqdm
import torch

from mtt.data import OnlineDataset
from mtt.data import generate_data


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if len(os.listdir(args.out_dir)) > 0:
        raise ValueError(f"Output directory {args.out_dir} is not empty.")

    online_dataset = OnlineDataset(n_steps=119, length=20, img_size=128, device="cuda")
    # start a generator that will yeild a simulation with 119 steps (see data.py)
    generator = generate_data(
        online_dataset,
        args.n_simulations,
    )
    for i, simulation in tqdm.tqdm(
        enumerate(generator),
        total=args.n_simulations,
        desc="Generating dataset",
        unit="sims",
    ):
        with open(os.path.join(args.out_dir, f"{i}.pt"), "wb") as f:
            torch.save(simulation, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", type=str, default="./data/train/")
    parser.add_argument("-n", "--n-simulations", type=int, default=100)
    args = parser.parse_args()
    main(args)

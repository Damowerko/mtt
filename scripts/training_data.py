import os
import argparse

import torch
import tqdm

from mtt.data import OnlineDataset
from mtt.data import generate_data


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if len(os.listdir(args.out_dir)) > 0:
        raise ValueError(f"Output directory {args.out_dir} is not empty.")

    online_dataset = OnlineDataset(n_steps=119, length=20, img_size=128, device="cuda")
    for i in tqdm.trange(
        args.n_files, desc="Generating dataset", unit="files", position=0
    ):
        data = generate_data(
            online_dataset,
            args.simulations_per_file,
            tqdm_kwargs=dict(
                desc=f"Generating file", unit="sims", position=1, leave=False
            ),
        )
        with open(os.path.join(args.out_dir, f"{i}.pt"), "wb") as f:
            torch.save(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", type=str, default="./data/train/")
    parser.add_argument("-n", "--n-files", type=int, default=100)
    parser.add_argument("-s", "--simulations-per-file", type=int, default=100)

    args = parser.parse_args()
    main(args)

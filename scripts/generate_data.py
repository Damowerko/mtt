import argparse
import os
import pickle
from functools import partial

import torch
import tqdm

from mtt.data import OnlineDataset, generate_vectors, stack_images
from mtt.simulator import Simulator


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if len(os.listdir(args.out_dir)) > 0:
        raise ValueError(f"Output directory {args.out_dir} is not empty.")

    online_dataset = OnlineDataset(
        n_steps=119,
        length=20,
        img_size=128 * args.scale,
        device="cuda",
        init_simulator=partial(Simulator, window_width=1000 * args.scale),
    )
    # start a generator that will yeild a simulation with 119 steps (see data.py)
    vectors_list = list(
        generate_vectors(
            online_dataset,
            args.n_simulations,
        )
    )
    # save the simulation data
    with open(os.path.join(args.out_dir, "simulations.pkl"), "wb") as f:
        pickle.dump(vectors_list, f)

    if args.no_images:
        return
    for i, simulation in tqdm.tqdm(
        enumerate(vectors_list),
        total=args.n_simulations,
        desc="Generating images",
        unit="image",
    ):
        images = [online_dataset.vector_to_image(v) for v in simulation]
        with open(os.path.join(args.out_dir, f"{i}.pt"), "wb") as f:
            torch.save(stack_images(images), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", type=str, default="./data/train/")
    parser.add_argument("-n", "--n-simulations", type=int, default=100)
    parser.add_argument(
        "-s", "--scale", type=int, default=1, help="Scale of the simulation in km."
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't generate images. Only save the simulation data.",
    )
    args = parser.parse_args()
    main(args)

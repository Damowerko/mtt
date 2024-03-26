import argparse
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import tqdm

from mtt.data.image import OnlineImageDataset, stack_images, to_image
from mtt.data.sim import SimulationStep, Simulator
from mtt.data.sparse import vector_to_df
from mtt.data.utils import parallel_rollout


def filter_simulation(
    simulation: list[SimulationStep], window_width: int, min_window_width: int = 0
):
    if min_window_width:
        # the number of measurements within each sample
        n_measurements = [len(s.measurements) for s in simulation]
        # the worst case number of measurements within a 20 step window
        n_measurements_min = (
            np.lib.stride_tricks.sliding_window_view(
                n_measurements, window_shape=window_width
            )
            .sum(-1)
            .min()
        )
        if n_measurements_min < min_window_width:
            return False
    return True


def main(args):
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    online_dataset = OnlineImageDataset(
        n_steps=args.n_steps,
        length=20,
        img_size=128 * args.scale,
        device="cuda",
        init_simulator=partial(Simulator, window_width=1000 * args.scale),
    )

    # check if simulation data already exists
    if (out_dir / "simulations.pkl").exists():
        print("Simulation data already exists. Loading from disk...")
        with (out_dir / "simulations.pkl").open("rb") as f:
            vectors_list = pickle.load(f)
    else:
        vectors_list = parallel_rollout(
            online_dataset.sim_generator,
            n_rollouts=args.n_simulations,
            tqdm_kwargs={"desc": "Generating simulation data", "unit": "simulation"},
        )

        # save the simulation data
        with (out_dir / "simulations.pkl").open("wb") as f:
            print("Dumping simulation data pickle to disk...")
            pickle.dump(vectors_list, f)

    if not args.no_parquet:
        # save the dataframes to parquet files
        if not any(
            [
                (out_dir / "targets.parquet").exists(),
                (out_dir / "measurements.parquet").exists(),
                (out_dir / "sensors.parquet").exists(),
            ]
        ):
            print("Converting to dataframes and saving to parquet files...")
            # convert the simulation data to parquet files
            df_targets, df_measurements, df_sensors = vector_to_df(
                vectors_list,
                tqdm_kwargs={"desc": "Converting to dataframes", "unit": "simulation"},
            )
            df_targets.to_parquet(out_dir / "targets.parquet")
            df_measurements.to_parquet(out_dir / "measurements.parquet")
            df_sensors.to_parquet(out_dir / "sensors.parquet")
        else:
            print("Parquet files already exist. Will not overwrite.")

    if not args.no_images:
        images_dir = out_dir / "images"
        if images_dir.exists():
            print("Images already exist. Will not overwrite.")
        else:
            images_dir.mkdir()

            for i, simulation in tqdm.tqdm(
                enumerate(vectors_list),
                total=args.n_simulations,
                desc="Generating images",
                unit="image",
            ):
                images = [to_image(v) for v in simulation]
                with (images_dir / f"{i}.pt").open("wb") as f:
                    torch.save(stack_images(images), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", type=str, default="./data/train/")
    parser.add_argument("-n", "--n-simulations", type=int, default=100)
    parser.add_argument(
        "--n_steps", type=int, default=119, help="Number of steps in the simulation."
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=1, help="Scale of the simulation in km."
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Don't save data to parquet dataframes.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't generate images. Only save the simulation data.",
    )
    args = parser.parse_args()
    main(args)

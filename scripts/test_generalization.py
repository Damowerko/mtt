import argparse
import os
from time import time
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from mtt.data import StackedImageData, build_test_datapipe, collate_fn
from mtt.models.convolutional import Conv2dCoder, load_model
from mtt.peaks import find_peaks
from mtt.utils import compute_ospa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        type=str,
        default="./models/e7ivqipk.ckpt",
        help="The uri of the model to test. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/test",
        help="The directory containing the test data. Should have folders called 1km, 2km, 3km etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/out/generalization",
        help="The directory to save the results.",
    )
    parser.add_argument(
        "--max-scale",
        type=int,
        default=5,
        help="The maximum scale to test on. The data directory should have folders called 1km, 2km, 3km etc.",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Test runtime instead of performance.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model, name = load_model(args.model_uri)
    output_filename = f"{name}_runtime.csv" if args.runtime else f"{name}.csv"

    results: List[pd.DataFrame] = []
    for scale in range(1, args.max_scale + 1):
        print(f"Testing model on {scale}km")
        test_fn = test_runtime if args.runtime else test_model
        result = test_fn(
            model,
            os.path.join(args.data_dir, f"{scale}km", "simulations.pkl"),
            scale=scale,
        )
        # data in data_dir is organized by scale in folders: 1km, 2km, 3km etc.
        result["scale"] = scale
        results.append(result)
        # save results after each scale to avoid losing everything if something goes wrong
        df = pd.concat(results)
        df.to_csv(os.path.join(args.output_dir, output_filename), index=False)


def test_runtime(model: Conv2dCoder, data_path: str, scale=1):
    model = model.cuda().eval()
    t_start = time()
    datapipe = build_test_datapipe(
        data_path,
        unbatch=False,
        vector_to_image_kwargs=dict(device="cuda", img_size=128 * scale),
    )
    # get the next data sample
    simulation = next(iter(datapipe))
    dt_load_data = time() - t_start

    simulation_iter = iter(simulation)
    times = []
    while True:
        t_start = time()
        data: StackedImageData | None = next(simulation_iter, None)
        if data is None:
            break
        t_data = time()

        with torch.no_grad():
            output = model.forward(data.sensor_images.cuda().unsqueeze(0))
        # sync to measure the time to finish computation
        torch.cuda.synchronize()
        t_forward = time()

        positions_estimates = find_peaks(
            output[-1, -1].cpu().numpy(),
            width=data.info[-1]["window"],
            method="kmeans",
        ).means
        t_peaks = time()

        times += [
            dict(
                data=t_data - t_start,
                forward=t_forward - t_data,
                peaks=t_peaks - t_forward,
            )
        ]
    df = pd.DataFrame(times)
    df["load"] = dt_load_data / len(df)
    return df


def test_model(model: Conv2dCoder, data_path: str, scale: int = 1):
    model = model.cuda()
    datapipe = build_test_datapipe(
        data_path,
        unbatch=False,
        vector_to_image_kwargs=dict(device="cuda", img_size=128 * scale),
    )
    result: List[Dict] = []
    for simulation_idx, simulation in enumerate(tqdm(datapipe, total=len(datapipe))):
        for step_idx, data in enumerate(simulation):
            data: StackedImageData = data
            with torch.no_grad():
                output = model.forward(data.sensor_images.cuda().unsqueeze(0))

            mse = (output[-1] - data.target_images[-1]).pow(2).mean().item()
            cardinality_target = model.cardinality_from_image(
                data.target_images[-1]
            ).item()
            cardinality_output = model.cardinality_from_image(output).item()
            cardinality_truth = len(data.info[-1]["target_positions"])

            positions_truth = data.info[-1]["target_positions"]
            positions_estimates = find_peaks(
                output[-1, -1].cpu().numpy(),
                width=data.info[-1]["window"],
                method="kmeans",
            ).means
            ospa = compute_ospa(positions_truth, positions_estimates, 500)

            result.append(
                dict(
                    simulation_idx=simulation_idx,
                    step_idx=step_idx,
                    mse=mse,
                    cardinality_target=cardinality_target,
                    cardinality_output=cardinality_output,
                    cardinality_truth=cardinality_truth,
                    ospa=ospa,
                )
            )
    return pd.DataFrame(result)


if __name__ == "__main__":
    main()

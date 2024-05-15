import argparse
import os
import pickle
from pathlib import Path
from time import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mtt.data.image import stack_images, to_image
from mtt.data.sim import SimulationStep
from mtt.data.sparse import SparseData, SparseDataset
from mtt.models.convolutional import Conv2dCoder
from mtt.models.kernel import RKHSBase
from mtt.models.sparse import SparseBase, SparseLabel
from mtt.models.utils import load_model
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

    model, name, params = load_model(args.model_uri)
    output_filename = f"{name}_runtime.csv" if args.runtime else f"{name}.csv"

    results: List[pd.DataFrame] = []
    for scale in range(1, args.max_scale + 1):
        print(f"Testing model on {scale}km")
        test_fn = test_runtime if args.runtime else test_model
        result = test_fn(
            model,
            os.path.join(args.data_dir, f"{scale}km"),
            scale=scale,
        )
        # data in data_dir is organized by scale in folders: 1km, 2km, 3km etc.
        result["scale"] = scale
        results.append(result)
        # save results after each scale to avoid losing everything if something goes wrong
        df = pd.concat(results)
        df.to_csv(os.path.join(args.output_dir, output_filename), index=False)


def test_runtime(model, data_path: str, scale: int = 1):
    if isinstance(model, Conv2dCoder):
        return test_cnn_runtime(model, data_path, scale=scale)
    else:
        raise ValueError(f"Model type {type(model)} not supported.")


@torch.no_grad()
def test_cnn_runtime(model: Conv2dCoder, data_path: str, scale=1):
    with open(Path(data_path) / "simulations.pkl", "rb") as f:
        simulations: list[list[SimulationStep]] = pickle.load(f)

    n_simulations = len(simulations)
    n_steps = len(simulations[0])

    times = []
    for sim_idx in tqdm(range(n_simulations)):
        for start_idx in range(n_steps - model.length + 1):
            stop_idx = start_idx + model.length
            t_start = time()
            images = [
                to_image(sim, img_size=128 * scale, device="cuda")
                for sim in simulations[sim_idx][start_idx:stop_idx]
            ]
            data = stack_images(images)
            t_data = time()

            # sync to measure the time to finish computation
            output = model.forward(data.sensor_images.cuda().unsqueeze(0))
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
    return df


def test_model(model, data_path: str, scale: int = 1):
    if isinstance(model, Conv2dCoder):
        return test_conv_model(model, data_path, scale=scale)
    elif isinstance(model, RKHSBase):
        return test_knn_model(model, data_path, scale=scale)


@torch.no_grad()
def test_knn_model(model: RKHSBase, data_path: str, scale=1):
    model = model.cuda()
    # update n_samples
    model.n_samples *= scale**2

    dataset = SparseDataset(data_path, length=model.input_length)
    result: List[Dict] = []
    indices = np.random.permutation(len(dataset))[:10]
    for idx in tqdm(indices):
        sim_idx, step_idx = dataset.parse_index(idx)
        data = dataset[idx]
        data = SparseData(*(t.cuda() for t in data))
        label = SparseLabel.from_sparse_data(data, model.input_length)
        label = SparseLabel(*(t.cuda() for t in label))
        output = model.forward(model.forward_input(data))
        # batch size should be one
        assert output.batch.shape[0] == 1
        ospa = model.ospa(label, output).item()
        # "kernel" loss means the energy of the difference between the RKHS functions ||f - g||^2
        # divide by scale**2 to get the per unit area loss
        mse = model.loss(label, output).item() / scale**2
        # TODO: add cardinality to result dict
        result.append(
            dict(
                simulation_idx=sim_idx,
                step_idx=step_idx,
                mse=mse,
                ospa=ospa,
            )
        )
    return pd.DataFrame(result)


@torch.no_grad()
def test_conv_model(model: Conv2dCoder, data_path: str, scale=1):
    model = model.cuda()
    with open(Path(data_path) / "simulations.pkl", "rb") as f:
        simulations: list[list[SimulationStep]] = pickle.load(f)

    n_simulations = len(simulations)
    n_steps = len(simulations[0])

    result: List[Dict] = []
    for sim_idx in tqdm(range(n_simulations)):
        images = [
            to_image(sim, img_size=128 * scale, device="cuda")
            for sim in simulations[sim_idx]
        ]
        for step_idx in range(n_steps - model.length + 1):
            data = stack_images(images[step_idx : step_idx + model.length])

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
                    simulation_idx=sim_idx,
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

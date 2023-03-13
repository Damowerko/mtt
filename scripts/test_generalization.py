import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from mtt.data import StackedImageData, build_test_datapipe, collate_fn
from mtt.models import Conv2dCoder, load_model
from mtt.peaks import find_peaks
from mtt.utils import compute_ospa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        type=str,
        default="./models/58c6fd8a.ckpt",
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
        default="./out/generalization",
        help="The directory to save the results.",
    )

    args = parser.parse_args()

    model, name = load_model(args.model_uri)
    results: List[pd.DataFrame] = []
    for scale in range(1, 2):
        print(f"Testing model on {scale}km")
        result = test_model(
            model,
            os.path.join(args.data_dir, f"{scale}km", "simulations.pkl"),
            scale=scale,
        )
        # data in data_dir is organized by scale in folders: 1km, 2km, 3km etc.
        result["scale"] = scale
        results.append(result)
    df = pd.concat(results)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, f"{name}.csv"), index=False)


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
                output = model.forward(data.sensor_images.cuda())

            mse = (output[-1] - data.target_images[-1]).pow(2).mean().item()
            cardinality_target = model.cardinality_from_image(
                data.target_images[-1]
            ).item()
            cardinality_output = model.cardinality_from_image(output).item()
            cardinality_truth = len(data.info[-1]["target_positions"])

            targets_truth = data.info[-1]["target_positions"]
            # targets_gmm = find_peaks(
            #     output[-1].cpu().numpy(), width=data.info[-1]["window"], model="gmm"
            # ).means
            targets_kmeans = find_peaks(
                output[-1].cpu().numpy(), width=data.info[-1]["window"], model="kmeans"
            ).means
            # ospa_gmm = compute_ospa(targets_truth, targets_gmm, 500)
            ospa_kmeans = compute_ospa(targets_truth, targets_kmeans, 500)

            result.append(
                dict(
                    simulation_idx=simulation_idx,
                    step_idx=step_idx,
                    mse=mse,
                    cardinality_target=cardinality_target,
                    cardinality_output=cardinality_output,
                    cardinality_truth=cardinality_truth,
                    # ospa_gmm=ospa_gmm,
                    ospa_kmeans=ospa_kmeans,
                )
            )
    return pd.DataFrame(result)


if __name__ == "__main__":
    main()

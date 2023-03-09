import argparse
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

import wandb
from mtt.data import StackedImageData, build_test_datapipe
from mtt.models import Conv2dCoder


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
    for scale_dir in os.listdir(args.data_dir):
        print(f"Testing model on {scale_dir}")
        result = test_model(
            model, os.path.join(args.data_dir, scale_dir, "simulations.pkl")
        )
        # data in data_dir is organized by scale in folders: 1km, 2km, 3km etc.
        result["scale"] = int(scale_dir.split("km")[0])
        results.append(result)
    df = pd.concat(results)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, f"{name}.csv"), index=False)


def test_model(model: Conv2dCoder, data_path: str):
    model = model.cuda()
    datapipe = build_test_datapipe(
        data_path, unbatch=False, vector_to_image_kwargs={"device": "cuda"}
    )
    result: List[Dict] = []
    for simulation_idx, simulation in enumerate(tqdm(datapipe, total=len(datapipe))):
        for step_idx, data in enumerate(simulation):
            assert isinstance(data, StackedImageData)

            output = model.forward(data.sensor_images.cuda())
            mse = model.loss(output, data.target_images[-1:].cuda()).item()

            cardinality_target = model.cardinality_from_image(
                data.target_images[-1:]
            ).item()
            cardinality_output = model.cardinality_from_image(output[-1:]).item()
            cardinality_truth = len(data.info[-1]["target_positions"])

            result.append(
                dict(
                    simulation_idx=simulation_idx,
                    step_idx=step_idx,
                    mse=mse,
                    cardinality_target=cardinality_target,
                    cardinality_output=cardinality_output,
                    cardinality_truth=cardinality_truth,
                )
            )
    return pd.DataFrame(result)


if __name__ == "__main__":
    main()

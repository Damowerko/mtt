import argparse
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import pandas as pd
import torch
import wandb
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from tqdm import tqdm

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

    datapipe = build_test_datapipe(data_path)
    rs = MultiProcessingReadingService(num_workers=10)
    dataloader = DataLoader2(datapipe, reading_service=rs)

    result: List[Dict] = []
    for simulation_idx, simulation in enumerate(tqdm(dataloader, total=len(datapipe))):
        for step_idx, data in enumerate(simulation):
            output = model(data.sensor_images.cuda())
            mse = torch.mean((output - data.target_images.cuda()) ** 2).item()
            result.append(
                dict(
                    simulation_idx=simulation_idx,
                    step_idx=step_idx,
                    mse=mse,
                )
            )
    return pd.DataFrame(result)


def load_model(uri: str) -> Tuple[Conv2dCoder, str]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
    """
    if uri.startswith("wandb://"):
        user, project, run_id = uri[len("wandb://") :].split("/")

        # Download the model from wandb to temporary directory
        with TemporaryDirectory() as tmpdir:
            api = wandb.Api()
            artifact = api.artifact(
                f"{user}/{project}/model-{run_id}:best_k", type="model"
            )
            artifact.download(root=tmpdir)
            model = Conv2dCoder.load_from_checkpoint(f"{tmpdir}/model.ckpt")
            name = run_id
    else:
        model = Conv2dCoder.load_from_checkpoint(uri)
        name = os.path.basename(uri).split(".")[0]
    return model, name


if __name__ == "__main__":
    main()

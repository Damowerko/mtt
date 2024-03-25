from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import yaml

from mtt.models.convolutional import EncoderDecoder
from mtt.models.sparse import SparseBase


def load_model(uri: str) -> Tuple[SparseBase | EncoderDecoder, str, dict]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
    """
    with TemporaryDirectory() as tmpdir:
        if uri.startswith("wandb://"):
            import wandb

            user, project, run_id = uri[len("wandb://") :].split("/")

            # Download the model from wandb to temporary directory
            api = wandb.Api()
            artifact = api.artifact(
                f"{user}/{project}/model-{run_id}:best", type="model"
            )
            artifact.download(root=tmpdir)

            uri = f"{tmpdir}/model.ckpt"
            name = run_id
            params = api.run(f"{user}/{project}/{run_id}").config
        else:
            path = Path(uri)
            name = path.stem
            params = yaml.safe_load(path.with_suffix(".yaml").read_text())

        model_class = get_model_class(params["model"])

        try:
            model = model_class.load_from_checkpoint(uri)
        except RuntimeError:
            model = model_class.load_from_checkpoint(uri, deprecated_api=True)
        return model, name, params


def get_model_class(name: str):
    if name == "conv2d":
        from mtt.models.convolutional import Conv2dCoder

        return Conv2dCoder
    elif name == "st":
        from mtt.models.transformer import SpatialTransformer

        return SpatialTransformer
    elif name == "knn":
        from mtt.models.kernel import KNN

        return KNN
    else:
        raise ValueError(f"Unknown model: {name}.")

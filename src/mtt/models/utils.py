from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import yaml

from mtt.models import EncoderDecoder, SparseBase


def load_model(uri: str) -> Tuple[SparseBase | EncoderDecoder, str]:
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

        if params.get("model") == "conv2d":
            from mtt.models import Conv2dCoder

            model_class = Conv2dCoder
        elif params.get("model") == "st":
            from mtt.models import SpatialTransformer

            model_class = SpatialTransformer
        elif params.get("model") == "knn":
            from mtt.models import KNN

            model_class = KNN

        try:
            model = model_class.load_from_checkpoint(uri)
        except RuntimeError:
            model = model_class.load_from_checkpoint(uri, deprecated_api=True)
        return model, name

# Multi Target Tracking
Multi target tracking with a fully convolutional encoder-decoder.

## Installation

1. Install (poetry 1.12)[https://python-poetry.org/docs/master/#installing-with-the-official-installer]. You can use the official installer or the pipx installer, whichever you prefer. If you use pipx you can install poetry using the following command.

`pipx install git+https://github.com/python-poetry/poetry.git`

2. By default poetry will automatically create a virtualenv for this project. If you do not want this behaviour run the following. I recommend `pyenv` and/or `conda` for managing your environments.

`poetry config virtualenvs.create false --local`

3. `poetry install` and all pip packages will install. This might take a while (up to ~10 minutes). If it's taking long it might also be a networking issue (sometimes pypi.org resolves to an IPv6 address, which may cause problems, try seeing if `pip install numpy` fails).

## Models

Model weights are available on my [Google Drive](), since they are too large for `git-lfs`. Download the contents of the `models` directory and place them in the `models` directory.

Alternatively, you can download them using `gdown`.

```
gdown --folder --id 17jW5YQpbT6i6gV2smHl69ezC5P8VhjAj
mv mtt-models/ models/
```
FROM damowerko/torchcps:latest

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# install mtt package in editable mode
COPY src src
COPY scripts scripts
RUN poetry install --only-root
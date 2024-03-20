FROM damowerko/torchcps:latest

# clone repository into container
RUN cd /home/$USER && git clone https://github.com/Damowerko/mtt

# set working directory to repository directory
WORKDIR /home/$USER/mtt

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# update with changed source code
COPY src src
COPY scripts scripts
RUN poetry install --only-root
FROM damowerko/torchcps:latest

# set working directory to repository directory
RUN mkdir /home/$USER/mtt
WORKDIR /home/$USER/mtt
# fix git ownership
RUN git config --global --add safe.directory /home/$USER/mtt

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# copy the rest of the repository
COPY . .
RUN poetry install --only-root
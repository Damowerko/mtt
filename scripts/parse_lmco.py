from argparse import ArgumentParser
from glob import glob

import pandas as pd
from tqdm import tqdm

from mtt.simulator import position_image
from mtt.utils import compute_ospa

parser = ArgumentParser()
parser.add_argument("filter_name", type=str)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/sim_data/",
    help="Directory where data is stored.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="data/out/",
    help="Directory where output is stored.",
)
args = parser.parse_args()

filter_name = args.filter_name
cols = ["scale", "simulation_idx", "step_idx", "posx_m", "posy_m", "exist_prob"]

df_truth = pd.read_csv(
    f"{args.data_dir}/{args.filter_name}_truth.csv", low_memory=False
)
df_estimate = pd.read_csv(
    f"{args.data_dir}/{args.filter_name}_estimates.csv", low_memory=False
)

# Parse Sensor Data
# sensor_files = list(
#     glob(f"{args.data_dir}/{args.filter_name}/**/sensor_states_*.csv", recursive=True)
# )
# df_sensor_list = []
# for sensor_file in tqdm(sensor_files, desc="Parsing Sensor Data"):
#     df = pd.read_csv(sensor_file, low_memory=False)
#     # Can get scale and simulation_idx from the path
#     df["simulation_idx"] = int(sensor_file.split("/")[-2])
#     df["scale"] = float(sensor_file.split("/")[-3].split("k")[0].split("_")[-1])
#     df_sensor_list.append(df)
# df_sensor = pd.concat(df_sensor_list, axis=0, ignore_index=True)


df_combined = (
    pd.concat(
        [df_truth, df_estimate], axis=0, keys=["truth", "estimate"]
    )
    .reset_index(level=0, names=["type"])
    .rename(columns={"time": "step_idx"})
)

# Convert several columns to integers
integer_columns = ["scale", "step_idx"]
df_combined[integer_columns] = df_combined[integer_columns].astype(int)

# Get the number of simulations, steps, and scales
n_scales = df_combined["scale"].unique().size
n_simulations = df_combined["simulation_idx"].max() + 1
n_steps = df_combined["step_idx"].max() + 1


def to_image(df: pd.DataFrame, scale: int):
    # get the positions of the targets
    # ignore targets with nan values for position
    na_positions = df[["posx_m", "posy_m"]].isna().any(axis=1)
    positions = df[["posx_m", "posy_m"]].to_numpy()[~na_positions]
    weights = df["exist_prob"].fillna(1.0).to_numpy()[~na_positions]

    # LMCO uses a convention of (0,0) being the corner of the image
    offset = 2000.0 + 1000.0 * scale / 2
    # subtract the offset to get the correct position in our coordinates
    positions -= offset

    assert len(positions) == len(weights)
    return position_image(
        window_width=1000 * scale,
        size=128 * scale,
        sigma=10.0,
        target_positions=positions,
        device="cuda",
        weights=weights,
    )



results = []
groupby_columns = ["simulation_idx", "step_idx", "scale"]
gb = df_combined.groupby(groupby_columns, as_index=False, sort=True)
for (idx, df) in tqdm(gb, total=len(gb), desc="Computing OSPA"):
    # comptue OSPA at each time-step for each simulation
    simulation_idx, step_idx, scale = idx
    truth = df[df["type"] == "truth"]
    filter = df[df["type"] == "estimate"]
    # sensor = df[df["type"] == "sensor"]

    truth_positions = truth[["posx_m", "posy_m"]].dropna().to_numpy()
    filter_positions = (
        filter
            # .query("exist_prob >= 0.5")
            [["posx_m", "posy_m"]]
            .dropna()
            .to_numpy()
    )
    ospa = compute_ospa(truth_positions, filter_positions, 500, 2)
    ospa1 = compute_ospa(truth_positions, filter_positions, 500, 1)

    truth_image = to_image(truth, scale)
    filter_image = to_image(filter, scale)
    mse = (truth_image - filter_image).pow(2).mean().item()
    results.append(
        dict(
            scale=scale,
            simulation_idx=simulation_idx,
            step_idx=step_idx,
            ospa=ospa,
            ospa1=ospa1,
            mse=mse,
            filter=args.filter_name,
            cardinality_truth=len(truth_positions),
            cardinality_estimate=filter["exist_prob"].sum(),
            # n_sensors=len(sensor["sensor_id"].unique()),
        )
    )
df_results = pd.DataFrame(results)
df_results.to_csv(f"{args.output_dir}/{args.filter_name}_summary.csv", index=False)

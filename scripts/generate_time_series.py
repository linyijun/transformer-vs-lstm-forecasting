import random

import pandas as pd
from tqdm import tqdm
import numpy as np
from uuid import uuid4
import argparse


def get_init_df():

    date_rng = pd.date_range(start="2015-01-01", end="2020-01-01", freq="D")

    dataframe = pd.DataFrame(date_rng, columns=["timestamp"])

    dataframe["index"] = range(dataframe.shape[0])

    dataframe["article"] = uuid4().hex

    return dataframe


def set_amplitude(dataframe):

    max_step = random.randint(90, 365)
    max_amplitude = random.uniform(0.1, 1)
    offset = random.uniform(-1, 1)

    phase = random.randint(-1000, 1000)

    amplitude = (
        dataframe["index"]
        .apply(lambda x: max_amplitude * (x % max_step + phase) / max_step + offset)
        .values
    )

    if random.random() < 0.5:
        amplitude = amplitude[::-1]

    dataframe["amplitude"] = amplitude

    return dataframe


def set_offset(dataframe):

    max_step = random.randint(15, 45)
    max_offset = random.uniform(-1, 1)
    base_offset = random.uniform(-1, 1)

    phase = random.randint(-1000, 1000)

    offset = (
        dataframe["index"]
        .apply(
            lambda x: max_offset * np.cos(x * 2 * np.pi / max_step + phase)
            + base_offset
        )
        .values
    )

    if random.random() < 0.5:
        offset = offset[::-1]

    dataframe["offset"] = offset

    return dataframe


def generate_time_series(dataframe, periods):

    clip_val = random.uniform(0.3, 1)

    period = random.choice(periods)

    phase = random.randint(-1000, 1000)

    dataframe["views"] = dataframe.apply(
        lambda x: np.clip(
            np.cos(x["index"] * 2 * np.pi / period + phase), -clip_val, clip_val
        )
        * x["amplitude"]
        + x["offset"],
        axis=1,
    ) + np.random.normal(
        0, dataframe["amplitude"].abs().max() / 10, size=(dataframe.shape[0],)
    )

    dataframe["period"] = dataframe["index"] % period
    dataframe["norm_period"] = dataframe["period"] / period
    dataframe["max_period"] = period
    return dataframe


def generate_df(periods):
    dataframe = get_init_df()
    dataframe = set_amplitude(dataframe)
    dataframe = set_offset(dataframe)
    dataframe = generate_time_series(dataframe, periods)
    return dataframe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/data.csv")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--periods", type=int, default=[7, 14, 28], nargs='+')

    args = parser.parse_args()

    dataframes = []

    for _ in tqdm(range(args.num_samples)):
        df = generate_df(args.periods)
        dataframes.append(df)

    all_data = pd.concat(dataframes, ignore_index=True)

    all_data.to_csv(args.data_path, index=False)
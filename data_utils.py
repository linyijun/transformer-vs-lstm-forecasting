import random
import numpy as np
import pandas as pd

import torch


def split_df(
    df: pd.DataFrame, split: str, history_size: int = 120, horizon_size: int = 30
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param history_size:
    :param horizon_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(horizon_size + 1, df.shape[0] - (horizon_size + history_size) * 2)
    elif split == "val": 
        end_index = df.shape[0] - (horizon_size + history_size)
    elif split == "test":
        end_index = df.shape[0]
    else:
        raise ValueError

    label_index = end_index - horizon_size
    start_index = max(0, label_index - history_size)

    history = df[start_index:label_index]
    targets = df[label_index:end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 120):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df):
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        groups, 
        grp_by, 
        split, 
        features, 
        target, 
        seq_len=120, 
        horizon=30,
        use_periodic_as_feat=True,
    ):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target
        self.target_lag_1 = f"{self.target}_lag_1"
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_periodic_as_feat = use_periodic_as_feat

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split, history_size=self.seq_len, horizon_size=self.horizon)

        selected_features = self.features
        if self.use_periodic_as_feat:
            selected_features += ['norm_period'] 
        
        src = src[[self.target] + selected_features + ['period']]
        src = df_to_np(src)

        trg_in = trg[[self.target_lag_1] + selected_features + ['period']]
        trg_in = np.array(trg_in)
        
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out
from collections import defaultdict
import shutil
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import os
from tqdm import tqdm

# to ignore warnings
warnings.filterwarnings("ignore")


def spherical_to_cartesian(horizontal_column, vertical_column):
    # each column is a pandas series
    # first convert to radians
    horizontal = np.radians(horizontal_column)
    vertical = np.radians(vertical_column)
    x = np.sin(horizontal)
    z = np.cos(vertical) * (1 - x**2)
    y = np.tan(vertical) * z
    return round(x, 3), round(y, 3), round(z, 3)


def preprocess_df(
    df, subsampling=True, openamount_placeholder=1, intersect_placeholder=0
):
    # remove columns xT, yT, zT
    df = df.drop(["n", "x", "y", "xT", "yT", "zT"], axis=1)
    # if subsampling, keep 1/4 of the data
    if subsampling:
        df = df.iloc[::4]
        # reset index
        df = df.reset_index(drop=True)

    # add a blink column for both left and right eyes.  set to 1 if the values are all nans
    df["blink_left"] = (
        df[["lx", "ly", "clx", "cly", "clz"]].isna().all(axis=1).astype(int)
    )
    df["blink_right"] = (
        df[["rx", "ry", "crx", "cry", "crz"]].isna().all(axis=1).astype(int)
    )

    # convert spherical to cartesian for both left and right eyes
    df["lx"], df["ly"], df["lz"] = spherical_to_cartesian(df["lx"], df["ly"])
    df["rx"], df["ry"], df["rz"] = spherical_to_cartesian(df["rx"], df["ry"])
    # add openamount for both left and right eyes and set all values to 1
    df["openamount_left"] = openamount_placeholder
    df["openamount_right"] = openamount_placeholder
    # add three columns for intersection with useful, relevant, and puzzle, and set all to 0
    df["intersectuseful"] = intersect_placeholder
    df["intersectrelevant"] = intersect_placeholder
    df["intersectpuzzle"] = intersect_placeholder

    # reorder columns
    df = df[
        [
            "openamount_left",
            "openamount_right",
            "blink_left",
            "blink_right",
            "intersectuseful",
            "intersectrelevant",
            "intersectpuzzle",
            "clx",
            "cly",
            "clz",
            "crx",
            "cry",
            "crz",
            "lx",
            "ly",
            "lz",
            "rx",
            "ry",
            "rz",
        ]
    ]

    # fill nans with 0
    df = df.fillna(0)
    return df


def raw_eye_tracking_to_time_series(
    df,
    start_id=0,
    window_size=120,
    step_size=60,
):
    df = preprocess_df(df)
    num_samples = start_id
    train_dfs, val_dfs = [], []
    val_start, val_end, _, _ = get_train_val_test_by_time(
        len(df), no_test=True, val_ratio=0.1
    )
    if val_start == None:  # only train
        train_dfs.append(df)
    else:
        val_dfs.append(df.iloc[val_start:val_end])
        train_dfs.extend(
            [
                df.iloc[:val_start],
                df.iloc[val_end:],
            ]
        )

    train_final_df, train_label_df, num_samples = extract_from_dfs(
        source_dfs=train_dfs,
        window_size=window_size,
        step_size=step_size,
        num_samples=num_samples,
    )
    if val_dfs == []:
        return (
            (train_final_df, train_label_df),
            (None, None),
            (None, None),
            num_samples,
        )
    val_final_df, val_label_df, num_samples = extract_from_dfs(
        source_dfs=val_dfs,
        window_size=window_size,
        step_size=step_size,
        num_samples=num_samples,
    )
    return (
        (train_final_df, train_label_df),
        (val_final_df, val_label_df),
        num_samples,
    )


def extract_from_dfs(
    source_dfs,
    window_size,
    step_size,
    num_samples,
):
    target_df = pd.DataFrame()
    target_labels = []
    target_feature_indices = []
    start = num_samples
    # need to deal with label differently. Distance/ADHD/Trial comes from outer info, phase comes from raw data
    for source_df in source_dfs:
        index = 0
        feature_df = source_df
        while index < len(source_df) - window_size + 1:
            # skip not full window and window that span two phases
            if index + window_size > len(source_df):
                break
            ind = np.arange(index, index + window_size)
            target_labels.append(0)
            target_df = pd.concat([target_df, feature_df.iloc[ind]], ignore_index=True)
            target_feature_indices += [num_samples] * window_size
            num_samples += 1
            index += step_size

    label_df = pd.get_dummies(target_labels).astype("float32")

    # convert to one hot
    # label_df = pd.get_dummies(label_df[0])
    target_df["ts_index"] = target_feature_indices
    target_df.set_index("ts_index", inplace=True)
    label_df["ts_index"] = list(range(start, num_samples))
    label_df.set_index("ts_index", inplace=True)
    return target_df, label_df, num_samples


def convert_raw_data(
    read_root_path,
    save_root_path,
    use_task=["VRG"],
    window_size=120,
    step_size=60,
):
    np.random.seed(42)
    num_samples = 0
    dataframes_by_owner = {"train": {}, "val": {}}
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            if file.endswith(".csv"):
                if not any([task in file for task in use_task]):
                    continue
                owner = file[2:6]
                (train_x, train_y), (val_x, val_y), num_samples = (
                    raw_eye_tracking_to_time_series(
                        pd.read_csv(os.path.join(root, file)),
                        start_id=num_samples,
                        window_size=window_size,
                        step_size=step_size,
                    )
                )
                for split, x, y in zip(
                    ["train", "val"], [train_x, val_x], [train_y, val_y]
                ):
                    if owner not in dataframes_by_owner[split]:
                        dataframes_by_owner[split][owner] = [x, y]
                    else:
                        dataframes_by_owner[split][owner][0] = pd.concat(
                            [dataframes_by_owner[split][owner][0], x]
                        )
                        dataframes_by_owner[split][owner][1] = pd.concat(
                            [dataframes_by_owner[split][owner][1], y]
                        )
    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    for split in ["train", "val"]:
        for owner, (x, y) in dataframes_by_owner[split].items():
            if not os.path.exists(os.path.join(save_root_path, f"{split}/{owner}")):
                os.makedirs(os.path.join(save_root_path, f"{split}/{owner}"))
            x.to_csv(os.path.join(save_root_path, f"{split}/{owner}/feature_df.csv"))
            y.to_csv(os.path.join(save_root_path, f"{split}/{owner}/label_df.csv"))


def get_train_val_test_by_user(user_label_dct, threshold=1, no_test=False):
    """
    user_label_dct: a dictionary of user label distribution. Each item is in the format of {user(int): label_dist(np.array)}
    """

    def find_valid_split(
        user_label_dct, used_indices, target_dist, sim_threshold=0.9, max_iter=10
    ):
        for i in range(max_iter):
            chosen_indices = np.random.choice(
                used_indices, size=4 if no_test else 3, replace=False
            )
            chosen_label_dist = np.sum(
                [user_label_dct[i] for i in chosen_indices], axis=0
            )
            # ensure no zeros:
            if (
                np.any(chosen_label_dist == 0)
                and i <= max_iter // 2
                and threshold >= 0.5
            ):
                continue
            chosen_label_dist = chosen_label_dist / np.linalg.norm(chosen_label_dist)
            # do a cosine similarity check
            if np.dot(chosen_label_dist, target_dist) >= sim_threshold:
                return chosen_indices
        return None

    full_label_dist = np.sum(list(user_label_dct.values()), axis=0)
    user_indices = np.array(list(user_label_dct.keys()))

    # get the overall label distribution

    full_label_dist = full_label_dist / np.linalg.norm(full_label_dist)

    # choose a random subset of three users for test, that has a roughly similar label distribution as the full dataset
    test_indices = None
    while threshold >= 0 and test_indices is None:
        threshold -= 0.05
        test_indices = find_valid_split(
            user_label_dct, user_indices, full_label_dist, sim_threshold=threshold
        )

    # remove the test indices from the user indices
    user_indices = np.setdiff1d(user_indices, test_indices)
    if no_test:
        return user_indices, test_indices, None

    # choose a random subset of three users for val, that has a roughly similar label distribution as the full dataset
    val_indices = None
    while threshold >= 0 and val_indices is None:
        val_indices = find_valid_split(
            user_label_dct, user_indices, full_label_dist, sim_threshold=threshold
        )
        threshold -= 0.05

    # remove the val indices from the user indices
    train_indices = np.setdiff1d(user_indices, val_indices)
    return train_indices, val_indices, test_indices


def get_train_val_test_by_time(
    total_time, val_ratio=0.1, test_ratio=0.1, minimum_time=90, no_test=True
):
    """
    find two non-overlapping intervals of length val_ratio and test_ratio in the total time. If < minimum_time, find length = minimum_time
    """
    if no_test:
        if total_time < minimum_time * 2:
            return None, None, None, None
        val_length = max(int(total_time * val_ratio), minimum_time)
        val_start = np.random.randint(0, total_time - val_length)
        return val_start, val_start + val_length, None, None

    if total_time < minimum_time * 3:
        return None, None, None, None
    # find the length of val and test
    val_length = max(int(total_time * val_ratio), minimum_time)
    test_length = max(int(total_time * test_ratio), minimum_time)
    # find the start of val and test
    val_start = np.random.randint(0, total_time - val_length - test_length)
    test_start = np.random.randint(val_start + val_length, total_time - test_length + 1)
    # randomly swap val and test
    if np.random.rand() > 0.5:
        val_start, test_start = test_start, val_start
    return val_start, val_start + val_length, test_start, test_start + test_length


if __name__ == "__main__":
    convert_raw_data(
        "datasets\pretrain\gazebasevr\data",
        "datasets\pretrain\cleaned",
        use_task=["VRG"],
    )

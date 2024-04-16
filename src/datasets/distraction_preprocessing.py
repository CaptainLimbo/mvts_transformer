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
from sklearn.model_selection import StratifiedShuffleSplit

# to ignore warnings
warnings.filterwarnings("ignore")
LABEL_NUM_DICT = {"distance": 3, "adhd": 2, "trial": 2, "phase": 2}
DISTANCES = {
    1: [
        8.46,
        18.36,
        21.33,
        45,
        5.1,
        16.25,
        3.7,
        8.53,
        6.48,
        57.96,
        7.8,
        27.21,
        7.67,
        25.69,
        6.88,
        20.34,
        14.21,
        33.81,
        32.29,
        34.14,
        29.39,
        52.75,
        17.97,
        15.72,
    ],
    2: [
        19.35,
        17.37,
        20,
        64,
        9.12,
        6.55,
        11.76,
        7.08,
        8.07,
        21.8,
        7.28,
        44.63,
        13.22,
        14.8,
        18.23,
        3.78,
        22.59,
        18.76,
        11.83,
        4.44,
        26.22,
        55.65,
        8.2,
        5.3,
    ],
    3: [
        4.97,
        29.98,
        60.93,
        3.51,
        9.59,
        5.43,
        1.53,
        13.88,
        9.12,
        43.38,
        7.14,
        32.75,
        32,
        32.95,
        22.26,
        15.66,
        28.26,
        14.05,
        39.35,
        17.7,
        36.25,
        62.88,
        13.28,
        8.6,
    ],
    4: [
        16.32,
        9.39,
        7.21,
        16,
        6.48,
        4.11,
        3.91,
        9.85,
        12.36,
        4.7,
        6.62,
        26.02,
        23.91,
        21.99,
        17.77,
        8,
        8.66,
        2.19,
        24.44,
        21.86,
        38.69,
        47.47,
        15.66,
        5.5,
    ],
    5: [
        2.59,
        8.86,
        53.15,
        13.55,
        8.66,
        18.1,
        3.98,
        2,
        13.81,
        34.34,
        5.96,
        31.76,
        8.99,
        20.74,
        25.03,
        11.5,
        11.17,
        11.1,
        14.54,
        24.5,
        27.01,
        52.22,
        9.12,
        4.5,
    ],
    6: [
        13.61,
        8.46,
        61,
        74,
        13.88,
        5.03,
        8.2,
        2.92,
        10.31,
        68.79,
        8.2,
        34.34,
        5.3,
        26.81,
        11.04,
        19.49,
        11.37,
        35.59,
        30.9,
        29.25,
        34.8,
        33.41,
        20.08,
        8.86,
    ],
}
ADHD_SCORES = [3, 4, 2, 0, 2, 1, 3, 4, 3, 0, 3, 0, 2, 0, 2, 4, 3, 6, 4, 3, 1, 1, 2, 3]
CHOSEN_COLUMNS = [
    "GazeDir",
    "Unnamed: 13",
    "Unnamed: 14",
    "HitInfoTarget",
    "Label",
]


# train, val, test index:
TRIAL_TRAIN_INDICES = [2, 14, 6, 3, 12, 21, 4, 5, 19, 17, 23, 24, 8, 11, 15, 20, 7]
TRIAL_VAL_INDICES = [1, 18, 9]
TRIAL_TEST_INDICES = [16, 10, 13]

ADHD_TRAIN_INDICES = [2, 5, 8, 17, 19, 23, 21, 14, 4, 12, 24, 6, 15, 10, 13, 20, 9]
ADHD_VAL_INDICES = [1, 11, 18]
ADHD_TEST_INDICES = [3, 7, 16]


def grouping_distance(distance):
    # <10: 0; 10-50: 1; >50: 2
    if distance <= 10:
        return 0
    elif distance <= 30:
        return 1
    else:
        return 2


def grouping_ADHD(score):
    return 0 if score <= 3 else 1


# def grouping_trial(trial):
#     return 1 if trial in [3, 4, 5, 6] else 0
def grouping_trial(trial, trial_number=None):
    if not trial_number or len(trial_number) == 6:
        # default: predict all distraction
        return 1 if trial in [3, 4, 5, 6] else 0
    else:
        return 1 if trial == trial_number[1] else 0


def grouping(name, value, trial_number=None):
    if name == "distance":
        return grouping_distance(value)
    elif name == "adhd":
        return grouping_ADHD(value)
    elif name == "trial":
        return grouping_trial(value, trial_number)


def split_columns_and_save(df, col, split_num=3):
    if split_num == 3:
        feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
            feature_df[col].str.strip("()").str.split("|", expand=True)
        )
    elif split_num == 4:
        feature_df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
            feature_df[col].str.strip("()").str.split("|", expand=True)
        )
    feature_df[f"{col}_x"] = pd.to_numeric(feature_df[f"{col}_x"])
    feature_df[f"{col}_y"] = pd.to_numeric(feature_df[f"{col}_y"])
    feature_df[f"{col}_z"] = pd.to_numeric(feature_df[f"{col}_z"])
    if split_num == 4:
        feature_df[f"{col}_o"] = pd.to_numeric(feature_df[f"{col}_o"])
    feature_df = feature_df.drop(col, axis=1)
    return df


def preprocess_target_to_int(df, use_one_hot=True, smooth_targets=True, trial_number=1):
    target_to_roi = {
        "Catheter_Middle": 1,
        "lateral_ventricle": 2,
        "third_ventricle": 2,
        "R_foramen_of_monro": 2,
        "L_foramen_of_monro": 2,
        "cerebral_aqueduct": 2,
        "Target_Distance": 3 if trial_number not in [1, 6] else 0,
        "Tool_Angle": 4 if trial_number not in [1, 6] else 0,
        "HeartRate": 5 if trial_number in [3, 5, 6] else 0,
    }

    def convert_zeros(array, sublist_length=5):
        result = np.empty_like(array)
        for i, x in enumerate(array):
            if x != 0:
                result[i] = x
            else:
                neighbor = array[
                    max(0, i - sublist_length // 2) : min(
                        len(array), i + sublist_length // 2 + 1
                    )
                ]
                max_value = np.bincount(neighbor).argmax()
                if np.count_nonzero(neighbor == max_value) >= sublist_length // 2 + 1:
                    result[i] = max_value
                else:
                    result[i] = 0
        return result

    # map df['target'] using the dict into a column of integers
    df["target"] = df["target"].map(target_to_roi).fillna(0).astype("int64")
    # clean the data. 0's in between the same target should be filled with the same value
    if smooth_targets:
        df["target"] = convert_zeros(df["target"].to_numpy())
    # change to one hot
    if not use_one_hot:
        return df
    target_numpy = list(df["target"].to_numpy())
    target_one_hot = pd.get_dummies(target_numpy + list(range(6))).astype("float64")
    # drop the added ones
    target_one_hot = target_one_hot.iloc[: len(df)]
    target_one_hot.columns = [
        "other",
        "catheter",
        "ventricle",
        "distance",
        "angle",
        "heart_rate",
    ]
    # put it back to df
    df = pd.concat([df, target_one_hot], axis=1)
    df.drop("target", axis=1, inplace=True)

    return df


def raw_eye_tracking_to_time_series(
    input_dataframe,
    start_id=0,
    window_size=150,
    step_size=150,
    label=None,
    use_phase=[4, 5],
    label_name="distance",
    time_split=False,
    no_test=True,
    smooth_targets=True,
    trial_number=1,
):
    """
    conversion of the raw eye tracking data to multivariate time series.
    Input csv is of the the following format:
        each row is a single timestamp with columns for all variates, including a label indicating whether a mistake was made
    We want to convert it to two pandas dataframe, one for features `feature_df` and another for labels `label_df`; and also obtain a mapping `id_mapping` from a index to the entries in the dataframe for extracting the features and labels. `feature_df[id_mapping[i]]` should be the 1-th time-series sample, which should be of the shape (seq_length, feat_dim), while `label_df[id_mapping[i]]` should be the corresponding label, which should be of the shape (num_labels, ) which should be (2, ) in this case where my label is binary.
    Each data sample should have a seq length of 60, meaning that it should span 60 timestamps (rows). We would like to use a sliding window with a step size of 10 rows to go over the input csv to extract all time-series. If in a window of 60 timestamps, a value "1" in the mistake column occurs, that sample would have a corresponding label of "1" in `label_df`, otherwise "0". Write me the python code using pandas and any other packages necessary to implement this method.
    """
    # Load the input CSV file
    df = input_dataframe

    # in the feature_df, don't need the last three columns
    # need to split columns where the values are strings (x|y|z) into three columns for x, y, z
    df = df.rename(
        columns={
            "HitInfoTarget": "target",
            "GazeDir": "gaze_x",
            "Unnamed: 13": "gaze_y",
            "Unnamed: 14": "gaze_z",
            "Label": "phase",
        }
    )
    # filter phase
    df = df[df["phase"].isin(use_phase)]

    if "GazeOrigin" in df.columns:
        df = df.rename(
            columns={
                "GazeOrigin": "origin_x",
                "Unnamed: 10": "origin_y",
                "Unnamed: 11": "origin_z",
            }
        )
    if "HeadPosition" in df.columns:
        df = split_columns_and_save(df, "HeadPosition", 3)
    if "HeadOrientation" in df.columns:
        df = split_columns_and_save(df, "HeadOrientation", 4)
    df = preprocess_target_to_int(
        df, use_one_hot=False, smooth_targets=smooth_targets, trial_number=trial_number
    )

    dfs_by_phase = [df[df["phase"] == phase] for phase in use_phase]
    num_samples = start_id
    if time_split:
        lengths_by_phase = [len(dfs_by_phase[i]) for i in range(len(dfs_by_phase))]
        train_dfs, val_dfs, test_dfs = [], [], []
        for i in range(len(dfs_by_phase)):
            val_start, val_end, test_start, test_end = get_train_val_test_by_time(
                lengths_by_phase[i], no_test=no_test, val_ratio=0.2 if no_test else 0.1
            )
            if val_start == None:  # only train
                train_dfs.append(dfs_by_phase[i])
            elif test_start == None:
                val_dfs.append(dfs_by_phase[i].iloc[val_start:val_end])
                train_dfs.extend(
                    [
                        dfs_by_phase[i].iloc[:val_start],
                        dfs_by_phase[i].iloc[val_end:],
                    ]
                )
            else:
                val_dfs.append(dfs_by_phase[i].iloc[val_start:val_end])
                test_dfs.append(dfs_by_phase[i].iloc[test_start:test_end])
                train_dfs.extend(
                    [
                        dfs_by_phase[i].iloc[: min(val_start, test_start)],
                        dfs_by_phase[i].iloc[
                            min(val_end, test_end) : max(val_start, test_start)
                        ],
                        dfs_by_phase[i].iloc[max(val_end, test_end) :],
                    ]
                )

        train_final_df, train_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=train_dfs,
            window_size=window_size,
            step_size=step_size,
            label=label,
            label_name=label_name,
            num_samples=num_samples,
        )
        if val_dfs == []:
            return (
                (train_final_df, train_label_df),
                (None, None),
                (None, None),
                num_samples,
            )
        val_final_df, val_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=val_dfs,
            window_size=window_size,
            step_size=step_size,
            label=label,
            label_name=label_name,
            num_samples=num_samples,
        )
        if test_dfs == []:
            return (
                (train_final_df, train_label_df),
                (val_final_df, val_label_df),
                (None, None),
                num_samples,
            )
        test_final_df, test_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=test_dfs,
            window_size=window_size,
            step_size=step_size,
            label=label,
            label_name=label_name,
            num_samples=num_samples,
        )
        return (
            (train_final_df, train_label_df),
            (val_final_df, val_label_df),
            (
                test_final_df,
                test_label_df,
            ),
            num_samples,
        )
    # else, not need to split train, test
    new_df, label_df, num_samples, labels = extract_from_dfs(
        source_dfs=[df],
        window_size=window_size,
        step_size=step_size,
        label=label,
        label_name=label_name,
        num_samples=start_id,
    )

    return (new_df, label_df, num_samples, len(labels))


def extract_from_dfs(
    source_dfs,
    window_size,
    step_size,
    label,
    label_name,
    num_samples,
):
    target_df = pd.DataFrame()
    target_labels = []
    target_feature_indices = []
    start = num_samples
    # need to deal with label differently. Distance/ADHD/Trial comes from outer info, phase comes from raw data
    for source_df in source_dfs:
        index = 0
        feature_df = source_df.drop("phase", axis=1)
        while index < len(source_df) - window_size + 1:
            # skip not full window and window that span two phases
            if index + window_size > len(source_df):
                break
            ind = np.arange(index, index + window_size)
            target_labels.append(label)
            target_df = pd.concat([target_df, feature_df.iloc[ind]], ignore_index=True)
            target_feature_indices += [num_samples] * window_size
            num_samples += 1
            index += step_size

    label_num = LABEL_NUM_DICT[label_name]

    label_df = pd.get_dummies(target_labels + list(range(label_num))).astype("float32")

    # drop the added ones
    label_df = label_df.iloc[: len(target_labels)]

    # convert to one hot
    # label_df = pd.get_dummies(label_df[0])
    target_df["ts_index"] = target_feature_indices
    target_df.set_index("ts_index", inplace=True)
    label_df["ts_index"] = list(range(start, num_samples))
    label_df.set_index("ts_index", inplace=True)
    return target_df, label_df, num_samples, target_labels


def convert_raw_data(
    read_root_path,
    save_root_path,
    label="trial",
    use_head=False,
    use_angle=False,
    use_gaze_origin=False,
    use_phase=[4, 5],
    use_trial=[3, 5],
    window_size=150,
    step_size=150,
    split_strategy="by_user",
    split_json_path="datasets/Distraction/split_val_only.json",
    no_test=True,
    smooth_targets=True,
):
    # traverse the root path to find all csv files
    # for each csv file, convert it to time series
    # files under the same second-level directory should be grouped together
    if not isinstance(use_trial, list):
        use_trial = eval(use_trial)
    if not isinstance(use_phase, list):
        use_phase = eval(use_phase)
    num_samples = 0
    np.random.seed(1337)
    if split_strategy != "by_time":
        dataframes_by_owner = {}
        labels_by_user = {}
    if split_strategy == "by_time":
        dataframes_by_owner = {"train": {}, "val": {}, "test": {}}
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            if file.endswith(".csv"):
                trial_number = int(file[-5])
                owner = int(root.split(os.path.sep)[-1])
                if use_trial and trial_number not in use_trial:
                    continue
                if label == "trial":
                    my_label = grouping("trial", trial_number, use_trial)
                elif label == "distance":
                    my_label = grouping("distance", DISTANCES[trial_number][owner - 1])
                elif label == "adhd":
                    my_label = grouping("adhd", ADHD_SCORES[owner - 1])
                else:
                    my_label = label
                use_cols = CHOSEN_COLUMNS
                if use_head:
                    use_cols += ["HeadPosition", "HeadOrientation"]
                if use_angle:
                    use_cols += ["Angle"]
                if use_gaze_origin:
                    use_cols += [
                        "GazeOrigin",
                        "Unnamed: 10",
                        "Unnamed: 11",
                    ]
                if split_strategy != "by_time":
                    x, y, num_samples, label_count = raw_eye_tracking_to_time_series(
                        pd.read_csv(
                            os.path.join(root, file), skiprows=1, usecols=CHOSEN_COLUMNS
                        ),
                        start_id=num_samples,
                        label=my_label,
                        use_phase=use_phase,
                        window_size=window_size,
                        step_size=step_size,
                        label_name=label,
                        no_test=no_test,
                        smooth_targets=smooth_targets,
                        trial_number=trial_number,
                    )

                    if owner not in labels_by_user:
                        labels_by_user[owner] = np.zeros(LABEL_NUM_DICT[label])
                    if label != "phase":
                        labels_by_user[owner][my_label] += label_count
                    else:
                        labels_by_user[owner] += label_count

                    if owner not in dataframes_by_owner:
                        dataframes_by_owner[owner] = [x, y]
                    else:
                        dataframes_by_owner[owner][0] = pd.concat(
                            [dataframes_by_owner[owner][0], x]
                        )
                        dataframes_by_owner[owner][1] = pd.concat(
                            [dataframes_by_owner[owner][1], y]
                        )
                else:
                    (
                        (train_x, train_y),
                        (val_x, val_y),
                        (test_x, test_y),
                        num_samples,
                    ) = raw_eye_tracking_to_time_series(
                        pd.read_csv(
                            os.path.join(root, file), skiprows=1, usecols=CHOSEN_COLUMNS
                        ),
                        start_id=num_samples,
                        label=my_label,
                        use_phase=use_phase,
                        window_size=window_size,
                        step_size=step_size,
                        label_name=label,
                        time_split=True,
                        no_test=no_test,
                        smooth_targets=smooth_targets,
                        trial_number=trial_number,
                    )
                    for split, x, y in zip(
                        ["train", "val", "test"],
                        [train_x, val_x, test_x],
                        [train_y, val_y, test_y],
                    ):
                        if x is None:
                            continue
                        if owner not in dataframes_by_owner[split]:
                            dataframes_by_owner[split][owner] = [x, y]
                        else:
                            dataframes_by_owner[split][owner][0] = pd.concat(
                                [dataframes_by_owner[split][owner][0], x]
                            )
                            dataframes_by_owner[split][owner][1] = pd.concat(
                                [dataframes_by_owner[split][owner][1], y]
                            )
    if os.path.exists(os.path.join(save_root_path, f"{label}")):
        shutil.rmtree(os.path.join(save_root_path, f"{label}"))
    if (
        split_strategy != "by_time"
    ):  # conduct the stratified split by user. Try to match label distribution among all splits!
        # first see if it's stored in the json file
        if os.path.exists(split_json_path):
            with open(split_json_path, "r") as f:
                split_dict = json.load(f)
            current_setting = (
                f"{label}_phase_{use_phase}_trial_{use_trial}_{window_size}_{step_size}"
            )
            if current_setting in split_dict:
                print("loading from json...")
                train_indices = split_dict[current_setting]["train"]
                val_indices = split_dict[current_setting]["val"]
                test_indices = split_dict[current_setting]["test"]
            else:
                train_indices, val_indices, test_indices = get_train_val_test_by_user(
                    labels_by_user,
                    threshold=0 if split_strategy == "by_sample" else 1,
                    no_test=no_test,
                )
                split_dict[current_setting] = {
                    "train": list(map(int, train_indices)),
                    "val": list(map(int, val_indices)),
                    "test": (
                        list(map(int, test_indices)) if test_indices is not None else []
                    ),
                }

                with open(split_json_path, "w") as f:
                    json.dump(split_dict, f)
        # if exists, delete

        for ssplit in ["train", "val", "test"]:
            if not os.path.exists(os.path.join(save_root_path, f"{label}/{ssplit}")):
                os.makedirs(os.path.join(save_root_path, f"{label}/{ssplit}"))
        if split_strategy != "by_time":
            for owner in dataframes_by_owner:
                x, y = dataframes_by_owner[owner]
                # find the split of this owner
                if int(owner) in train_indices:
                    split = "train"
                elif int(owner) in val_indices:
                    split = "val"
                elif int(owner) in test_indices:
                    split = "test"
                else:
                    raise ValueError("owner not in split")
                if not os.path.exists(
                    os.path.join(save_root_path, f"{label}/{split}/{owner}")
                ):
                    os.makedirs(
                        os.path.join(save_root_path, f"{label}/{split}/{owner}")
                    )
                x.to_csv(
                    os.path.join(
                        save_root_path, f"{label}/{split}/{owner}/feature_df.csv"
                    )
                )
                y.to_csv(
                    os.path.join(
                        save_root_path, f"{label}/{split}/{owner}/label_df.csv"
                    )
                )
    else:
        for split in ["train", "val", "test"]:
            for owner in dataframes_by_owner[split]:
                x, y = dataframes_by_owner[split][owner]
                if not os.path.exists(
                    os.path.join(save_root_path, f"{label}/{split}/{owner}")
                ):
                    os.makedirs(
                        os.path.join(save_root_path, f"{label}/{split}/{owner}")
                    )
                if x is not None:
                    x.to_csv(
                        os.path.join(
                            save_root_path, f"{label}/{split}/{owner}/feature_df.csv"
                        )
                    )
                    y.to_csv(
                        os.path.join(
                            save_root_path, f"{label}/{split}/{owner}/label_df.csv"
                        )
                    )


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
        "datasets/Raw/Distraction Study",
        "datasets/Distraction",
        label="distance",
        use_gaze_origin=False,
        window_size=90,
        step_size=90,
    )

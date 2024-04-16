import shutil
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

TIME_FORMAT = "%H:%M:%S:%f"
# to ignore warnings
warnings.filterwarnings("ignore")

USED_COLUMNS = [
    "LeftEyeOpenAmount",
    "RightEyeOpenAmount",
    "LeftEyeBlinking",
    "RightEyeBlinking",
    "LeftEyeCenter",
    "RightEyeCenter",
    "LeftEyeGazeDirection",
    "RightEyeGazeDirection",
    "IntersectWithUseful",
    "IntersectWithNormal",
    "IntersectWithPuzzle",
    "HintReceived",
    "Mistake",
    "FalseMistake",
    "DevilDistraction",
    "ArtificialDistraction",
    "AudioDistraction",
    "BlockingDistraction",
]


def convert_label_columns(df, unique_labels=True):
    # Initialize the Label column with 0
    df["Label"] = 0
    # Apply the rules in the provided order
    # Rule 1: If DevilDistraction has value 1, then Label should also have value 1
    df.loc[df["DevilDistraction"] == 1, "Label"] = 1

    # Rule 2: If Mistake has value 1, and in the following 300 rows, none of FalseMistake has value 1, then Label should have value 2 for the previous rows starting from the row that contains the most recent "1" in the column "HintReceived"
    # Use a rolling window of size 120 to check the condition
    df["FalseMistakeWindow"] = df["FalseMistake"].rolling(120, min_periods=1).sum()
    mistake_indices = df[df["Mistake"] == 1].index
    for mistake_index in mistake_indices:
        if df.loc[mistake_index, "FalseMistakeWindow"] == 0:
            hint_received_indices = (
                df.loc[:mistake_index, "HintReceived"]
                .loc[df["HintReceived"] == 1]
                .index
            )

            if len(hint_received_indices) > 0:
                label_index = (
                    hint_received_indices[-2]
                    if len(hint_received_indices) > 1
                    else hint_received_indices[-1] - 120  # 2s prior to a mistake
                )
                df.loc[label_index:mistake_index, "Label"] = 2 if unique_labels else 1
    # Drop the temporary column
    df.drop("FalseMistakeWindow", axis=1, inplace=True)

    # Rule 3: If AudioDistraction has value 1, then Label should have value 3 in the following 90 rows
    audio_distraction_indices = df[df["AudioDistraction"] == 1].index
    for audio_index in audio_distraction_indices:
        df.loc[audio_index : audio_index + 90, "Label"] = 3 if unique_labels else 1

    # df.loc[df["AudioDistraction"] == 1, "Label"] = 3 if unique_labels else 1

    # Rule 4: If BlockingDistraction has value 1, then Label should have value 4
    # df.loc[df["BlockingDistraction"] == 1, "Label"] = 4 if unique_labels else 1

    # Return the modified dataframe
    df.drop(
        [
            "DevilDistraction",
            "Mistake",
            "FalseMistake",
            "HintReceived",
            "AudioDistraction",
            "BlockingDistraction",
            "ArtificialDistraction",
        ],
        axis=1,
        inplace=True,
    )
    return df


def split_columns_and_save(df, col, split_num=3):
    if split_num == 3:
        df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
            df[col].str.strip("()").str.split("|", expand=True)
        )
    elif split_num == 4:
        df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
            df[col].str.strip("()").str.split("|", expand=True)
        )
    df[f"{col}_x"] = pd.to_numeric(df[f"{col}_x"])
    df[f"{col}_y"] = pd.to_numeric(df[f"{col}_y"])
    df[f"{col}_z"] = pd.to_numeric(df[f"{col}_z"])
    if split_num == 4:
        df[f"{col}_o"] = pd.to_numeric(df[f"{col}_o"])
    df = df.drop(col, axis=1)
    return df


def compute_angular_velocity(feature_df):
    feature_df["x"] = (
        feature_df["FixationPointX"]
        - (feature_df["LeftEyeCenterX"] + feature_df["RightEyeCenterX"]) / 2
    )
    feature_df["y"] = (
        feature_df["FixationPointY"]
        - (feature_df["LeftEyeCenterY"] + feature_df["RightEyeCenterY"]) / 2
    )
    feature_df["z"] = (
        feature_df["FixationPointZ"]
        - (feature_df["LeftEyeCenterZ"] + feature_df["RightEyeCenterZ"]) / 2
    )
    # Calculate the difference between consecutive rows to get Δx, Δy, Δz
    df = feature_df[["x", "y", "z"]].copy()

    df["x_next"] = df["x"].shift(-1)
    df["y_next"] = df["y"].shift(-1)
    df["z_next"] = df["z"].shift(-1)

    # Calculate the dot product and magnitude of vectors
    df["dot_product"] = (
        df["x"] * df["x_next"] + df["y"] * df["y_next"] + df["z"] * df["z_next"]
    )
    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2) * np.sqrt(
        df["x_next"] ** 2 + df["y_next"] ** 2 + df["z_next"] ** 2
    )

    # Calculate the cosine of the angle
    df["cos_theta"] = df["dot_product"] / df["magnitude"]
    df["cos_theta"] = np.clip(df["cos_theta"], -1, 1)
    # print(df["cos_theta"])

    # Calculate the angle in radians
    df["theta"] = np.arccos(df["cos_theta"])
    df["degree"] = np.degrees(df["theta"]).astype(float)
    # only keep 5 decimal places
    df["degree"] = df["degree"].round(5)

    # Calculate the angular velocity
    feature_df["angular_velocity"] = df["degree"]

    return feature_df


def read_and_plot(
    data_path,
    label_column,
    data_column,
    ylim,
    ylabel,
    additional_function=None,
    clean=False,
):
    df = pd.read_csv(data_path)
    # take absolute value
    if additional_function:
        df[data_column] = additional_function(df[data_column])

    if clean:  # clean rows where either "LeftEyeOpen" or "RightEyeOpen" is 0
        df = df.loc[(df["LeftEyeBlinking"] == 0) & (df["RightEyeBlinking"] == 0)]

    # Create the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=label_column,
        y=data_column,
        data=df,
        color="blue",
        showfliers=False,
        width=0.6,
    )
    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("Distraction", fontsize=14)
    plt.xticks(
        [0, 1, 2, 3, 4], ["None", "Visual", "Before Mistake", "After Audio", "Blocking"]
    )

    # Show the plot
    plt.savefig(f"plots/{ylabel}.png")
    # plt.show()


def read_all_time_data(root_path, pattern="withdigit"):
    proportions = [0, 0, 0, 0]
    count = 0
    for root, dirs, files in os.walk(root_path):

        for file in files:
            print(root)
            print(root.split(os.path.sep)[-1])
            if root.split(os.path.sep)[-1].startswith("02") and not root.split(os.path.sep)[-1].endswith("YX"):
                if file.endswith(".csv") and pattern in file:
                    prop = read_and_plot_time(os.path.join(root, file))
                    count += 1
                    for i in range(4):
                        proportions[i] += prop[i]
    for i in range(4):
        proportions[i] /= count
        
    print(proportions)


def read_and_plot_time(data_path):
    df = pd.read_csv(data_path)
    targets = [
        "IntersectWithUseful",
        "IntersectWithNormal",
        "IntersectWithPuzzle",
        "IntersectWithDevil",
    ]
    times = []
    df = df.loc[(df["LeftEyeBlinking"] == 0) & (df["RightEyeBlinking"] == 0)]
    start = datetime.strptime(df["RealTime"].iloc[0], TIME_FORMAT)
    end = datetime.strptime(df["RealTime"].iloc[-1], TIME_FORMAT)
    total_time = (end - start).total_seconds()
    total_frames = len(df)
    print("Total time: ", total_time)
    for target in targets:
        df_slice = df[df[target] == 1]
        num_chuncks, time_spent = calc_consecutive_time_spent(df_slice)
        times.append(time_spent / total_time)
        # print("Number of chunks spent on ", target, ": ", num_chuncks)
        # print(
        #     "Prop of frames spent on ",
        #     target,
        #     ": ",
        #     round((len(df_slice) - num_chuncks) / total_frames, 4),
        # )
        print(f"Prop of Time spent on {target}: {time_spent / total_time:.4f}")
    times[2] -= times[0]
    return times
    # plt.figure(figsize=(10, 6))
    # plt.bar(
    #     targets,
    #     times,
    #     color="blue",
    #     width=0.6,
    # )
    # plt.ylabel("Time Spent (ms)", fontsize=14)
    # plt.xlabel("Target", fontsize=14)
    # plt.xticks(targets)
    # plt.title(f"Time spent on different targets")
    # plt.show()


def calc_consecutive_time_spent(d):
    start = None
    end = None
    consecutive = []
    for row in d.index:
        # set start to first row in spliced df
        if start is None:
            start = row
            end = row
        # if curr row is the next one, continue to add
        elif row == end + 1:
            end = row
        # no consecutive, add this chunk to list
        else:
            consecutive.append((start, end))
            start = row
            end = row
    # add the last chunk
    if start != end:
        consecutive.append((start, end))

    # go through consectutive chunks and add up the times in microseconds
    tot_time = 0
    # print(consecutive)
    for span in consecutive:
        start_time = d.loc[span[0]]["RealTime"]
        end_time = d.loc[span[1]]["RealTime"]
        # print(start_time, end_time)
        duration = datetime.strptime(end_time, TIME_FORMAT) - datetime.strptime(
            start_time, TIME_FORMAT
        )
        tot_time += duration.total_seconds()
    return len(consecutive), tot_time


def read_and_compute(
    data_path,
    label_column,
    data_column,
):
    label_mapping = {
        0: "None",
        1: "Visual",
        2: "Before Mistake",
        3: "After Audio",
        4: "Blocking",
    }
    count_data = []
    df = pd.read_csv(data_path)
    df = convert_label_columns(df, unique_labels=True)
    # find the number of continuous blocks of > 5 rows where "data_column" is 1 for each label
    for label in df[label_column].unique():
        label_df = df[df[label_column] == label]
        data = label_df[data_column].values
        continuous_blocks = []
        start = None
        for i, d in enumerate(data):
            if d == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= 5:
                        continuous_blocks.append((start, i))
                    start = None
        if start is not None:
            if len(data) - start > 5:
                continuous_blocks.append((start, len(data)))

        block_counts = len(continuous_blocks)
        start = datetime.strptime(df["RealTime"].iloc[0], TIME_FORMAT)
        end = datetime.strptime(df["RealTime"].iloc[-1], TIME_FORMAT)
        total_time = (end - start).total_seconds()

        print(
            f"Number of {data_column} blocks during Distraction {label_mapping[label]}: {block_counts}"
        )
        print(
            f"Avg Number of {data_column} per minute during Distraction {label_mapping[label]}: {block_counts / total_time * 60:.4f}"
        )
        count_data.append(block_counts / len(label_df) * 3600)

    print("=====================================")
    # make a bar plot of count_data - label
    plt.figure(figsize=(10, 6))
    plt.bar(
        [label_mapping[label] for label in df[label_column].unique()],
        count_data,
        color="blue",
        width=0.6,
    )
    plt.ylabel(f"Avg Number of {data_column} per minute", fontsize=14)
    plt.xlabel("Distraction", fontsize=14)
    plt.xticks(
        ["None", "Visual", "Before Mistake", "After Audio", "Blocking"],
    )
    plt.savefig(f"plots/{data_column}.png")


def raw_eye_tracking_to_time_series(
    input_dataframe,
    start_id=0,
    window_size=120,
    step_size=60,
    mistake_attn_offset=60,
    minimum_overlap_required=0.75,
    split="time",
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

    columns_to_split = [
        # "FixationPoint",
        "LeftEyeCenter",
        "RightEyeCenter",
        "LeftEyeGazeDirection",
        "RightEyeGazeDirection",
    ]
    for col in columns_to_split:
        df = split_columns_and_save(df, col, split_num=3)

    feature_df = convert_label_columns(df, unique_labels=False)
    # fill the nan with 0
    feature_df.fillna(0, inplace=True)
    num_samples = start_id
    if split == "time":
        train_dfs, val_dfs, test_dfs = [], [], []
        val_start, val_end, test_start, test_end = get_train_val_test_by_time(
            len(feature_df),
            val_ratio=0.2,
            minimum_time=120,
            no_test=True,
        )
        if val_start == None:  # only train
            train_dfs.append(feature_df)
        elif test_start == None:
            val_dfs.append(feature_df.iloc[val_start:val_end])
            train_dfs.extend(
                [
                    feature_df.iloc[:val_start],
                    feature_df.iloc[val_end:],
                ]
            )
        else:
            val_dfs.append(feature_df.iloc[val_start:val_end])
            test_dfs.append(feature_df.iloc[test_start:test_end])
            train_dfs.extend(
                [
                    feature_df.iloc[: min(val_start, test_start)],
                    feature_df.iloc[
                        min(val_end, test_end) : max(val_start, test_start)
                    ],
                    feature_df.iloc[max(val_end, test_end) :],
                ]
            )
        train_final_df, train_label_df, num_samples, _ = extract_from_dfs(
            source_dfs=train_dfs,
            window_size=window_size,
            step_size=step_size,
            num_samples=num_samples,
            minimum_overlap_required=minimum_overlap_required,
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
            num_samples=num_samples,
            minimum_overlap_required=minimum_overlap_required,
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
            num_samples=num_samples,
            minimum_overlap_required=minimum_overlap_required,
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


def extract_from_dfs(
    source_dfs, window_size, step_size, num_samples, minimum_overlap_required=0.75
):
    new_df = pd.DataFrame()
    labels = []
    feature_indices = []
    inner_start = num_samples
    for feature_df in source_dfs:
        feature_df.reset_index(drop=True, inplace=True)
        indices = [
            np.arange(i, i + window_size)
            for i in range(0, len(feature_df) - window_size + 1, step_size)
        ]
        clean_df = feature_df.drop("Label", axis=1)
        for ind in indices:
            # first consider manual distraction, this would be hard coded ground truth
            if 1 in feature_df["Label"].iloc[ind].values:
                # if all are 1, then label is 1
                if np.all(feature_df["Label"].iloc[ind].values == 1):
                    labels.append(1)
                    # id_mapping[num_samples] = ind
                    new_df = pd.concat([new_df, clean_df.iloc[ind]], ignore_index=True)
                    feature_indices += [num_samples] * window_size
                    num_samples += 1
                    continue
                else:
                    # find the row number of starting 1 and ending 1 in the window
                    start = np.where(feature_df["Label"].iloc[ind].values == 1)[0][0]
                    end = np.where(feature_df["Label"].iloc[ind].values == 1)[0][-1]
                    if 0 < start < step_size and ind[-1] + start < len(
                        feature_df
                    ):  # check if we can extend it
                        if np.all(feature_df["Label"].iloc[ind + start].values == 1):
                            labels.append(1)
                            # id_mapping[num_samples] = ind + start
                            new_df = pd.concat(
                                [new_df, clean_df.iloc[ind + start]], ignore_index=True
                            )
                            feature_indices += [num_samples] * window_size
                            num_samples += 1
                            continue
                    elif (window_size - step_size - 1 < end < window_size - 1) and ind[
                        0
                    ] - (
                        window_size - end - 1
                    ) >= 0:  # check if we can move forward the window a little bit
                        if np.all(
                            feature_df["Label"]
                            .iloc[ind - (window_size - end - 1)]
                            .values
                            == 1
                        ):
                            labels.append(1)
                            # id_mapping[num_samples] = ind - (window_size - end - 1)
                            new_df = pd.concat(
                                [new_df, clean_df.iloc[ind - (window_size - end - 1)]],
                                ignore_index=True,
                            )
                            feature_indices += [num_samples] * window_size
                            num_samples += 1
                            continue
                    elif (
                        np.sum(feature_df["Label"].iloc[ind].values) / window_size
                        > minimum_overlap_required
                    ):
                        labels.append(1)
                        # id_mapping[num_samples] = ind
                        new_df = pd.concat(
                            [new_df, clean_df.iloc[ind]], ignore_index=True
                        )
                        feature_indices += [num_samples] * window_size
                        num_samples += 1
                        continue
            else:
                labels.append(0)
                # id_mapping[num_samples] = ind
                feature_indices += [num_samples] * window_size
                new_df = pd.concat([new_df, clean_df.iloc[ind]], ignore_index=True)
                num_samples += 1
    label_df = pd.get_dummies(labels).astype("float32")
    # new_df.drop("Label", axis=1, inplace=True)

    # convert to one hot
    # label_df = pd.get_dummies(label_df[0])
    new_df["ts_index"] = feature_indices
    new_df.set_index("ts_index", inplace=True)
    label_df["ts_index"] = list(range(inner_start, num_samples))
    label_df.set_index("ts_index", inplace=True)
    return new_df, label_df, num_samples, labels


def convert_raw_data(
    read_root_path,
    save_root_path,
    normalize=True,
    split_strategy="time",
    window_size=120,
    step_size=60,
    minimum_overlap_required=0.75,
):
    # traverse the root path to find all csv files
    # for each csv file, convert it to time series
    # files under the same second-level directory should be grouped together
    if split_strategy == "time":
        dataframes_by_owner = {"train": {}, "val": {}, "test": {}}
    num_samples = 0
    open_amount_distribution = {}
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            # first round, only to get the distribution of open amount
            if normalize and file.endswith(".csv") and file.startswith("easy"):
                owner = root.split(os.path.sep)[-1]
                if owner.endswith("ZQ"):
                    continue
                reference_df = pd.read_csv(os.path.join(root, file))
                scale_left, scale_right = MinMaxScaler(), MinMaxScaler()
                scale_left.fit(reference_df["LeftEyeOpenAmount"].values.reshape(-1, 1))
                scale_right.fit(
                    reference_df["RightEyeOpenAmount"].values.reshape(-1, 1)
                )
                open_amount_distribution[owner] = (
                    scale_left,
                    scale_right,
                )
            if file.endswith(".csv") and not file.startswith("easy"):
                owner = root.split(os.path.sep)[-1]
                if owner.endswith("ZQ"):
                    continue
                dataframe = pd.read_csv(os.path.join(root, file), usecols=USED_COLUMNS)
                if normalize:
                    # round to 3 decimal places
                    dataframe["LeftEyeOpenAmount"] = (
                        open_amount_distribution[owner][0]
                        .transform(dataframe["LeftEyeOpenAmount"].values.reshape(-1, 1))
                        .round(3)
                    )
                    dataframe["RightEyeOpenAmount"] = (
                        open_amount_distribution[owner][1]
                        .transform(
                            dataframe["RightEyeOpenAmount"].values.reshape(-1, 1)
                        )
                        .round(3)
                    )
                if split_strategy == "time":
                    (
                        (train_x, train_y),
                        (val_x, val_y),
                        (test_x, test_y),
                        num_samples,
                    ) = raw_eye_tracking_to_time_series(
                        dataframe,
                        start_id=num_samples,
                        window_size=window_size,
                        step_size=step_size,
                        minimum_overlap_required=minimum_overlap_required,
                        split=split_strategy,
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

                    # x, y, num_samples = raw_eye_tracking_to_time_series(
                    #     dataframe,
                    #     start_id=num_samples,
                    #     window_size=window_size,
                    #     step_size=step_size,
                    #     minimum_overlap_required=minimum_overlap_required,
                    #     split=split_strategy,
                    # )
                    # if owner not in dataframes_by_owner:
                    #     dataframes_by_owner[owner] = [x, y]
                    # else:
                    #     dataframes_by_owner[owner][0] = pd.concat(
                    #         [dataframes_by_owner[owner][0], x]
                    #     )
                    #     dataframes_by_owner[owner][1] = pd.concat(
                    #         [dataframes_by_owner[owner][1], y]
                    #     )
    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    for ssplit in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(save_root_path, f"{ssplit}")):
            os.makedirs(os.path.join(save_root_path, f"{ssplit}"))
    if split_strategy == "time":
        for split in ["train", "val", "test"]:
            for owner in dataframes_by_owner[split]:
                x, y = dataframes_by_owner[split][owner]
                if not os.path.exists(os.path.join(save_root_path, f"{split}/{owner}")):
                    os.makedirs(os.path.join(save_root_path, f"{split}/{owner}"))
                if x is not None:
                    x.to_csv(
                        os.path.join(save_root_path, f"{split}/{owner}/feature_df.csv")
                    )
                    y.to_csv(
                        os.path.join(save_root_path, f"{split}/{owner}/label_df.csv")
                    )

    # for owner in dataframes_by_owner:
    #     x, y = dataframes_by_owner[owner]
    #     if not os.path.exists(os.path.join(save_root_path, f"{owner}")):
    #         os.makedirs(os.path.join(save_root_path, f"{owner}"))
    #     x.to_csv(os.path.join(save_root_path, f"{owner}/feature_df.csv"))
    #     y.to_csv(os.path.join(save_root_path, f"{owner}/label_df.csv"))


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


def plot_all_amount_data(read_root_path="datasets/raw", eye="Left", filter=None):
    column_name = f"{eye}EyeOpenAmount"
    easy_data_by_owner = {}
    hard_data_by_owner = {}
    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            if file.endswith(".csv"):
                owner = root.split("\\")[-1]
                if "easy" in file:
                    easy_data_by_owner[owner] = pd.read_csv(os.path.join(root, file))[
                        column_name
                    ].values
                elif "hard" in file:
                    hard_data_by_owner[owner] = np.concatenate(
                        (
                            easy_data_by_owner.get(owner, np.array([])),
                            pd.read_csv(os.path.join(root, file))[column_name].values,
                        )
                    )
    print(easy_data_by_owner.keys())
    data_to_plot = [
        [easy_data_by_owner[owner], hard_data_by_owner[owner]]
        for owner in easy_data_by_owner
        if owner in hard_data_by_owner
    ]
    print(data_to_plot[3][1])

    fig, ax = plt.subplots()
    start_pos = 1
    for group in data_to_plot:
        ax.boxplot(
            group,
            positions=[start_pos, start_pos + 1],
            widths=0.8,
            showfliers=False,
        )
        start_pos += 2

    # Set the x-axis labels
    ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9.5])
    ax.set_xticklabels(["User 1", "User 2", "User 3", "User 4", "User 5"])

    # Set the y-axis label
    ax.set_ylabel(f"{eye} Eye Open Amount", fontsize=14)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    read_all_time_data(
        "D:\Research\I3T\Projects\Eye Tracking\data\Fake User Study Data"
    )
    # read_and_plot_time(
    #     "D:\Research\I3T\Projects\Eye Tracking\data\Fake User Study Data\\02162024-1430-FY\hard-1418-withdigit_notdistracted.csv"
    # )
    # convert_raw_data("datasets\\raw\Fake User Study Data", "datasets\\ET_TS_Split_Time")
    # plot_all_amount_data(eye="Right")
    exit()
    """
    # split_columns_and_save(
    #     "datasets\\10-21-25-39-EyeTrackingData.csv",
    #     "datasets\\sudoku_1418_Sarah.csv",
    # )
    angular_velocity_f = lambda x: x.abs() * 60

    read_and_plot(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "angular_velocity",
        (0, 60),
        "Angular Velocity (degree per sec)",
        additional_function=angular_velocity_f,
        clean=True,
    )

    read_and_plot(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "LeftEyeOpenAmount",
        (0.4, 0.65),
        "Left Eye Open Amount",
        clean=True,
    )

    read_and_plot(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "RightEyeOpenAmount",
        (0.45, 0.7),
        "Right Eye Open Amount",
        clean=True,
    )

    read_and_compute(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "LeftEyeBlinking",
    )
    read_and_compute(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "RightEyeBlinking",
    )

    read_and_plot(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "LeftEyeBlinking",
        (0, 1.2),
        "Left Eye Blinking",
    )
    read_and_plot(
        "datasets\\sudoku_1418_Sarah_processed.csv",
        "Label",
        "RightEyeBlinking",
        (0, 1.2),
        "Right Eye Blinking",
    )
    """
    data_path = "D:\\Research\\I3T\\Projects\\Eye Tracking\\Attention Detection\\mvts_transformer\\datasets\\ET_TS\\Fake User Study Data\\12212023-1030-SE\\hard-1417-1.csv"
    df = pd.read_csv(data_path)
    feature_df, label_df, id_mapping = raw_eye_tracking_to_time_series(df)
    # save the processed data into csvs
    feature_df.to_csv(
        "D:\\Research\\I3T\\Projects\\Eye Tracking\\Attention Detection\\mvts_transformer\\datasets\\ET_TS\\feature_df.csv",
    )
    label_df.to_csv(
        "D:\\Research\\I3T\\Projects\\Eye Tracking\\Attention Detection\\mvts_transformer\\datasets\\ET_TS\\label_df.csv",
        index=False,
    )
    # save the mapping with pickle
    with open(
        "D:\\Research\\I3T\\Projects\\Eye Tracking\\Attention Detection\\mvts_transformer\\datasets\\ET_TS\\id_mapping.json",
        "wb",
    ) as f:
        pickle.dump(id_mapping, f, pickle.HIGHEST_PROTOCOL)

    # read the mapping
    with open(
        "D:\\Research\\I3T\\Projects\\Eye Tracking\\Attention Detection\\mvts_transformer\\datasets\\ET_TS\\id_mapping.json",
        "rb",
    ) as f:
        id_mapping = pickle.load(f)

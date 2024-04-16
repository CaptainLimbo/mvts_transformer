import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import os

# to ignore warnings
warnings.filterwarnings("ignore")


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

    df.loc[df["AudioDistraction"] == 1, "Label"] = 3 if unique_labels else 1

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


def split_columns_and_save(in_path, out_path):
    feature_df = pd.read_csv(in_path)
    columns_to_split = [
        "FixationPoint",
        "LeftEyeCenter",
        "RightEyeCenter",
        "LeftEyeGazeDirection",
        "RightEyeGazeDirection",
    ]
    for col in columns_to_split:
        feature_df[[f"{col}X", f"{col}Y", f"{col}Z"]] = (
            feature_df[col].str.strip("()").str.split("|", expand=True)
        )
        print()
        feature_df[f"{col}X"] = pd.to_numeric(feature_df[f"{col}X"])
        feature_df[f"{col}Y"] = pd.to_numeric(feature_df[f"{col}Y"])
        feature_df[f"{col}Z"] = pd.to_numeric(feature_df[f"{col}Z"])
        feature_df = feature_df.drop(col, axis=1)
    feature_df = convert_label_columns(feature_df)
    # compute gaze vector
    feature_df.to_csv(out_path, index=False)


def read_and_plot_2D_gaze(data_path, label=1, clean=True):
    distraction_label_type_dict = {
        0: "No Distraction",
        1: "Visual Distraction",
        2: "Mistake",
        3: "Audio Distraction",
    }
    df = pd.read_csv(data_path)
    # take absolute value

    if clean:  # clean rows where either "LeftEyeOpen" or "RightEyeOpen" is 0
        df = df.loc[(df["LeftEyeBlinking"] == 0) & (df["RightEyeBlinking"] == 0)]
    df = convert_label_columns(df, unique_labels=True)
    df = df.loc[df["Label"] == label]

    # split the two 2D columns
    columns_to_split = ["2DFixatedPoint", "2DFixatedPointFree"]
    gaze2D = {}
    for col in columns_to_split:
        df[[f"{col}X", f"{col}Y", f"{col}Z"]] = (
            df[col].str.strip("()").str.split("|", expand=True)
        )
        df[f"{col}X"] = pd.to_numeric(df[f"{col}X"])
        df[f"{col}Y"] = pd.to_numeric(df[f"{col}Y"])
        df[f"{col}Z"] = pd.to_numeric(df[f"{col}Z"])
        df = df.drop(col, axis=1)
        gaze2D[col] = df[[f"{col}X", f"{col}Y", f"{col}Z"]].to_numpy()
    _2dfixatedpoint = gaze2D["2DFixatedPoint"]
    _2dfixatedpointfree = gaze2D["2DFixatedPointFree"]

    combined_fixations = np.where(
        np.all(_2dfixatedpoint == [-100, -100, -100], axis=1, keepdims=True),
        _2dfixatedpointfree,
        _2dfixatedpoint,
    )

    combined_fixations = combined_fixations[
        (combined_fixations[:, 1] >= 10) & (combined_fixations[:, 1] <= 20)
    ]

    _2d_fixations = combined_fixations[:, [0, 2]]
    # plot it as a 2D graph, points connected by edges
    plt.plot(
        _2d_fixations[:, 0],
        _2d_fixations[:, 1],
        "-o",
        linestyle="-",
        # markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1,
    )
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.title(f"Gaze Path with {distraction_label_type_dict[label]}")
    plt.savefig(f"Gaze Path with {distraction_label_type_dict[label]}.png")
    plt.show()


if __name__ == "__main__":
    for label in (0, 1, 3):
        read_and_plot_2D_gaze(
            "datasets\\raw\\Fake User Study Data\\01042024-1700-ZQ\\hard-1418.csv",
            label,
        )

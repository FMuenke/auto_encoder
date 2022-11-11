import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from auto_encoder.util import load_dict


def load_model(model_folder):
    clf_results_file = os.path.join(model_folder, "classifier_results.csv")
    df = None
    if not os.path.isfile(os.path.join(model_folder, "training_history.pkl")):
        return None
    if os.path.isfile(clf_results_file):
        df = pd.read_csv(clf_results_file)

        opt = load_dict(os.path.join(model_folder, "opt.json"))
        if "task_difficulty" not in opt:
            opt["task_difficulty"] = 0.25
        for k in opt:
            if k == "input_shape":
                df[k] = str(opt[k])
            else:
                df[k] = opt[k]

        logs_df = pd.read_csv(os.path.join(model_folder, "logs.csv"))
        df["min_val_mse"] = logs_df["val_mse"].min()

    return df


def select_properties(df, properties):
    new_df = df
    for p in properties:
        if type(properties[p]) is list:
            new_df = new_df[new_df[p].isin(properties[p])]
        else:
            new_df = new_df[new_df[p] == properties[p]]
    return new_df


def main(args_):
    mf = args_.model

    data_frame = []
    for model_f in os.listdir(mf):
        model_df = load_model(os.path.join(mf, model_f))
        if model_df is None:
            continue
        data_frame.append(model_df)

    data_frame = pd.concat(data_frame, ignore_index=True)
    print(data_frame)
    print(data_frame.iloc[data_frame['Accuracy'].idxmax()])

    properties = {
        "clf": ["LR"],
        "n_labels": 2500,
        "depth": [2],
        "resolution": [16],
        # "embedding_size": 256,
        "drop_rate": 0.0,
        # "task": "reconstruction",
        # "task_difficulty": 0.25,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": "residual",
    }
    lr_data_frame = select_properties(data_frame, properties)
    print(lr_data_frame['Accuracy'].max())
    sns.lineplot(data=lr_data_frame, x="task_difficulty", y="Accuracy", hue="task")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        help="Folder containing all models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


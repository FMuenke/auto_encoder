import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from auto_encoder.util import load_dict


def load_model(model_folder):
    if not os.path.isdir(model_folder):
        return None
    clf_results_file = os.path.join(model_folder, "classifier_results.csv")
    df = None
    if not os.path.isfile(os.path.join(model_folder, "training_history.pkl")):
        return None
    if os.path.isfile(clf_results_file):
        df = pd.read_csv(clf_results_file)

        opt = load_dict(os.path.join(model_folder, "opt.json"))
        if "type" not in opt:
            return None
        if "asymmetrical" not in opt:
            opt["asymmetrical"] = False
        if "skip" not in opt:
            opt["skip"] = False
        if "asymmetrical" in opt["backbone"]:
            opt["backbone"] = opt["backbone"].replace("asymmetrical_", "")
            opt["asymmetrical"] = True
        if "skip" in opt["backbone"]:
            opt["backbone"] = opt["backbone"].replace("skip_", "")
            opt["skip"] = True
        if "task_difficulty" not in opt:
            opt["task_difficulty"] = 0.25
        for k in opt:
            if k == "input_shape":
                df[k] = str(opt[k])
            else:
                df[k] = opt[k]

        df["structure"] = "{}-{}".format(str(df["type"][0]), str(df["backbone"][0]))

        logs_df = pd.read_csv(os.path.join(model_folder, "logs.csv"))
        if "val_mse" not in logs_df:
            df["min_val_mse"] = logs_df["val_reconstruction_loss"].min()
        else:
            df["min_val_mse"] = logs_df["val_mse"].min()

        df["epochs"] = len(logs_df) - 128

    return df


def load_sub_folder(path):
    if not os.path.isdir(path):
        return None
    data_frame = []
    for model_f in os.listdir(path):
        model_df = load_model(os.path.join(path, model_f))
        if model_df is None:
            continue
        data_frame.append(model_df)
    return data_frame


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
        sub_df = load_sub_folder(os.path.join(mf, model_f))
        if sub_df is None:
            continue
        for df in sub_df:
            if df is None:
                continue
            data_frame.append(df)

    data_frame = pd.concat(data_frame, ignore_index=True)
    data_frame = data_frame.replace({"clf": "TF-MLP (2048, 1024) drp=0.75"}, "MLP")
    print(data_frame)
    print(data_frame.iloc[data_frame['Accuracy'].idxmax()])

    properties = {
        "type": ["autoencoder"],
        "clf": ["MLP"],
        "n_labels": 10000,
        # "depth": [2],
        # "resolution": [16],
        "embedding_size": [128],
        # "drop_rate": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "leaky_relu",
        # "backbone": ["residual"],
        "use_skip": True,
        "asymmetrical": False
    }

    sns.scatterplot(data=data_frame, x="epochs", y="Accuracy", hue="backbone")
    plt.show()

    lr_data_frame = select_properties(data_frame, properties)
    print(lr_data_frame)
    best_performer = lr_data_frame[lr_data_frame['Accuracy'] == lr_data_frame['Accuracy'].max()].to_dict()
    for k in best_performer:
        print(k, best_performer[k])
    g = sns.FacetGrid(lr_data_frame, col="resolution", row="depth", hue="skip")
    g.map(sns.lineplot, "drop_rate", "Accuracy")
    g.add_legend()

    # sns.catplot(data=lr_data_frame, x="drop_rate", y="Accuracy", hue="skip")

    # sns.lineplot(data=lr_data_frame, x="drop_rate", y="Accuracy", hue="backbone", markers=True, style="backbone")
    # sns.scatterplot(data=data_frame, x="min_val_mse", y="Accuracy", hue="backbone")
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


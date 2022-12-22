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
        if "embedding_noise" not in opt:
            opt["embedding_noise"] = 0.0
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

        df["epochs"] = len(logs_df)

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
    data_frame = data_frame.replace({"type": "variational-autoencoder"}, "vae")
    data_frame = data_frame.replace({"type": "autoencoder"}, "ae")
    print(data_frame)
    print(data_frame.iloc[data_frame['Accuracy'].idxmax()])
    plot_dropout_impact(data_frame, mf)
    plot_architecture_impact(data_frame, mf)
    plot_embedding_type_impact(data_frame, mf)
    plot_task_impact(data_frame, mf)
    plot_asymmetry_impact(data_frame, mf)
    plot_noise_impact(data_frame, mf)

    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        # "resolution": [16],
        # "embedding_size": [256],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual"],
        "skip": False,
        "asymmetrical": False,
    }

    lr_data_frame = select_properties(data_frame, properties)
    print(lr_data_frame)
    best_performer = lr_data_frame[lr_data_frame['Accuracy'] == lr_data_frame['Accuracy'].max()].to_dict()
    for k in best_performer:
        print(k, best_performer[k])
    # g = sns.FacetGrid(lr_data_frame, col="embedding_type", hue="resolution")
    # g.map(sns.lineplot, "embedding_size", "Accuracy")
    # g.add_legend()

    # sns.catplot(data=lr_data_frame, x="drop_rate", y="Accuracy", hue="skip")

    # sns.lineplot(data=lr_data_frame, x="resolution", y="Accuracy", hue="embedding_size", markers=True, style="embedding_size")
    sns.lineplot(data=lr_data_frame, x="min_val_mse", y="Accuracy", hue="resolution")
    plt.show()


def plot_task_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        # "task": "reconstruction",
        # "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual"],
        "skip": False,
        "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(
        data=lr_data_frame,
        x="task_difficulty",
        y="Accuracy",
        hue="task", markers=True, style="task"
    )
    plt.savefig(os.path.join(result_path, "task_impact.png"))
    plt.close()


def plot_embedding_type_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        # "embedding_type": "glob_avg",
        # "embedding_activation": "linear",
        "backbone": ["residual"],
        "skip": False,
        "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.catplot(
        data=lr_data_frame,
        x="embedding_type",
        y="Accuracy",
        hue="embedding_activation",
    )
    plt.savefig(os.path.join(result_path, "embedding_type_impact.png"))
    plt.close()


def plot_asymmetry_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        # "resolution": [16],
        # "embedding_size": [256],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual", "linear"],
        "skip": False,
        # "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    g = sns.FacetGrid(lr_data_frame, col="asymmetrical", row="backbone", hue="embedding_size")
    g.map(sns.lineplot, "resolution", "Accuracy")
    g.add_legend()
    plt.savefig(os.path.join(result_path, "asymmetry_impact.png"))
    plt.close()


def plot_noise_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        # "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual"],
        "skip": False,
        "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(
        data=lr_data_frame,
        x="embedding_noise",
        y="Accuracy",
    )
    plt.savefig(os.path.join(result_path, "embedding_noise_impact.png"))
    plt.close()


def plot_architecture_impact(data_frame, result_path):
    properties = {
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["linear", "residual"],
        "skip": False,
        "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    g = sns.FacetGrid(lr_data_frame, col="backbone", row="type", hue="embedding_size")
    g.map(sns.lineplot, "resolution", "Accuracy")
    g.add_legend()
    plt.savefig(os.path.join(result_path, "architecture_impact.png"))
    plt.close()


def plot_dropout_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual"],
        "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(
        data=lr_data_frame[lr_data_frame["skip"] == False],
        x="drop_rate",
        y="Accuracy",
        hue="dropout_structure",
        markers=True,
        style="dropout_structure"
    )
    plt.savefig(os.path.join(result_path, "dropout_impact_no_skip.png"))
    plt.close()

    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(
        data=lr_data_frame,
        x="drop_rate",
        y="Accuracy",
        hue="dropout_structure",
        markers=True,
        style="skip"
    )
    plt.savefig(os.path.join(result_path, "dropout_impact.png"))
    plt.close()

    sns.lineplot(
        data=lr_data_frame[lr_data_frame["skip"] == True],
        x="drop_rate",
        y="Accuracy",
        hue="dropout_structure",
        markers=True,
        style="dropout_structure"
    )
    plt.savefig(os.path.join(result_path, "dropout_impact_skipp.png"))
    plt.close()


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


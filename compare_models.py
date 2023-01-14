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
            opt["type"] = "CNN"
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
        if "scale" not in opt:
            opt["scale"] = 0
        if "embedding_noise" not in opt:
            opt["embedding_noise"] = 0.0
        for k in opt:
            if k == "input_shape":
                df[k] = str(opt[k])
            else:
                df[k] = opt[k]

        df["Label (Asym.)"] = "{}-Asym:{}".format(opt["backbone"], opt["asymmetrical"])
        df["Label (Depth.)"] = "{}-D{}".format(opt["backbone"], opt["depth"])

        df["structure"] = "{}-{}".format(str(df["type"][0]), str(df["backbone"][0]))
        if os.path.isfile(os.path.join(model_folder, "ae-weights-final.hdf5")):
            df["pretrained"] = True
        else:
            df["pretrained"] = False

        logs_df = pd.read_csv(os.path.join(model_folder, "logs.csv"))
        if "val_reconstruction_loss" in logs_df:
            df["min_val_mse"] = logs_df["val_reconstruction_loss"].min()
            df["val_mse"] = logs_df["val_reconstruction_loss"].iloc[-1]
        elif "val_mse" in logs_df:
            df["min_val_mse"] = logs_df["val_mse"].min()
            df["val_mse"] = logs_df["val_mse"].iloc[-1]
        else:
            df["min_val_mse"] = 0.0
            df["val_mse"] = 0.0

        df["epochs"] = len(logs_df)
        df["folder"] = os.path.basename(os.path.dirname(model_folder))
        df["tr-name"] = os.path.basename(model_folder)

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
    data_frame = data_frame.replace({"n_labels": 0}, 50000)
    # print(data_frame.iloc[data_frame['Accuracy'].idxmax()])
    plot_val_mse_impact(data_frame, mf)
    data_frame = data_frame[data_frame["folder"] != "AE.0"]
    plot_cnn_clf(data_frame, mf)
    plot_embedding_type_impact(data_frame, mf)
    data_frame = data_frame[data_frame["clf"] != "cnn"]

    count = 0
    for i, row in data_frame.sort_values("Accuracy", ascending=False).iterrows():
        count += 1
        if count > 15:
            break
        print("{:3}: {:10.3f} : {} / {}".format(count, row["Accuracy"], row["tr-name"], row["clf"]))

    count = 0
    print(" ")
    print("LR - CLASSIFIER:")
    for i, row in data_frame[data_frame["clf"] == "LR"].sort_values("Accuracy", ascending=False).iterrows():
        count += 1
        if count > 15:
            break
        print("{:3}: {:10.3f} : {} / {}".format(count, row["Accuracy"], row["tr-name"], row["clf"]))

    plot_dropout_impact(data_frame, mf)
    plot_architecture_impact(data_frame, mf)
    plot_task_impact(data_frame, mf)
    plot_asymmetry_impact(data_frame, mf)
    plot_noise_impact(data_frame, mf)

    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "scale": [0],
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
        # "skip": False,
        # "asymmetrical": True,
        # "pretrained": True,
        # "freeze": False
    }

    lr_data_frame = select_properties(data_frame, properties)
    # g = sns.FacetGrid(lr_data_frame, col="embedding_type", hue="resolution")
    # g.map(sns.lineplot, "embedding_size", "Accuracy")
    # g.add_legend()

    # sns.catplot(data=lr_data_frame, x="drop_rate", y="Accuracy", hue="skip")

    # sns.lineplot(
    # data=lr_data_frame,
    # x="resolution",
    # y="Accuracy",
    # hue="embedding_size",
    # markers=True,
    # style="embedding_size"
    # )
    sns.lineplot(data=lr_data_frame,
                 x="min_val_mse", y="Accuracy",
                 hue="embedding_size",
                 markers=True, style="asymmetrical"
                 )
    # plt.show()


def plot_cnn_clf(data_frame, result_path):
    properties = {"backbone": ["d-residual"]}
    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(data=lr_data_frame, x="scale", y="Accuracy", hue="freeze", markers=True, style="pretrained")
    plt.savefig(os.path.join(result_path, "cnn_clf_d_residual.png"))
    plt.close()

    properties = {
        "type": ["ae"],
        "clf": ["cnn"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "drop_rate": 0.0,
        "scale": 0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        # "task": "reconstruction",
        # "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual", "b-residual", "d-residual"],
        "skip": False,
        "asymmetrical": False,
        "freeze": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    # print(lr_data_frame)
    # sns.catplot(data=lr_data_frame, x="tr-name", y="Accuracy", hue="backbone")
    # plt.savefig(os.path.join(result_path, "cnn_clf_activation.png"))
    # plt.xticks(rotation=45)
    # plt.show()


def plot_val_mse_impact(data_frame, result_path):
    properties = {
        "folder": "AE.0",
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2, 4],
        "resolution": [16],
        "embedding_size": [256],
        "drop_rate": 0.0,
        "scale": 0,
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
    fig, ax = plt.subplots(2, figsize=(10, 12))
    sns.barplot(ax=ax[0], data=lr_data_frame[lr_data_frame["depth"] == 2], x="epochs", y="val_mse")
    sns.barplot(ax=ax[1], data=lr_data_frame[lr_data_frame["depth"] == 2], x="epochs", y="Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "val_mse.png"))
    plt.close()

    fig, ax = plt.subplots(2, figsize=(10, 12))
    sns.barplot(ax=ax[0], data=lr_data_frame[lr_data_frame["depth"] == 4], x="epochs", y="val_mse")
    sns.barplot(ax=ax[1], data=lr_data_frame[lr_data_frame["depth"] == 4], x="epochs", y="Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "val_mse_4.png"))
    plt.close()


def plot_task_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2],
        "resolution": [16],
        "embedding_size": [256],
        "scale": 0,
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
        # "clf": ["MLP", "cnn"],
        "n_labels": 10000,
        "depth": [2],
        "scale": 0,
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
        # "pretrained": True,
        # "freeze": False,
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.catplot(
        data=lr_data_frame[lr_data_frame["clf"] == "MLP"],
        x="embedding_type",
        y="Accuracy",
        hue="embedding_activation",
    )
    plt.savefig(os.path.join(result_path, "embedding_type_impact.png"))
    plt.close()

    sns.catplot(
        data=lr_data_frame[lr_data_frame["clf"] == "cnn"],
        x="embedding_type",
        y="Accuracy",
        hue="embedding_activation",
        col="freeze"
    )
    plt.savefig(os.path.join(result_path, "embedding_type_impact_cnn.png"))
    plt.close()


def plot_asymmetry_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2], "scale": 0,
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
    print(lr_data_frame)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.lineplot(ax=axs[0], data=lr_data_frame[lr_data_frame["backbone"] == "linear"],
                 x="resolution", y="Accuracy", style="Label (Asym.)", markers=True, hue="embedding_size")
    sns.lineplot(ax=axs[1], data=lr_data_frame[lr_data_frame["backbone"] == "residual"],
                 x="resolution", y="Accuracy", style="Label (Asym.)", markers=True, hue="embedding_size")
    plt.savefig(os.path.join(result_path, "asymmetry_impact.png"))
    plt.close()


def plot_noise_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2], "scale": 0, "resolution": [16],
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
        "depth": [2], # "scale": 0,
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

    sns.lineplot(
        data=lr_data_frame[lr_data_frame["type"] == "ae"],
        x="resolution", y="Accuracy", hue="embedding_size",
        style="backbone", markers=True,
    )
    plt.savefig(os.path.join(result_path, "architecture_impact_2.png"))
    plt.close()

    properties = {
        "clf": ["MLP"],
        "type": "ae",
        "n_labels": 10000,
        "depth": [1, 2, 4],
        "scale": 0,
        "drop_rate": 0.0,
        "dropout_structure": "general",
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["residual"],
        "skip": False,
        # "asymmetrical": False,
    }
    lr_data_frame = select_properties(data_frame, properties)

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    sns.lineplot(ax=axs[0], data=lr_data_frame[lr_data_frame["depth"] == 1],
                 x="resolution", y="Accuracy", style="asymmetrical", markers=True, hue="embedding_size")
    sns.lineplot(ax=axs[1], data=lr_data_frame[lr_data_frame["depth"] == 2],
                 x="resolution", y="Accuracy", style="asymmetrical", markers=True, hue="embedding_size")
    sns.lineplot(ax=axs[2], data=lr_data_frame[lr_data_frame["depth"] == 4],
                 x="resolution", y="Accuracy", style="asymmetrical", markers=True, hue="embedding_size")

    axs[0].set_title("Depth: 1")
    axs[1].set_title("Depth: 2")
    axs[2].set_title("Depth: 4")
    for ax in axs:
        ax.set_ylim([0.15, 0.35])
    # g = sns.FacetGrid(lr_data_frame, col="embedding_size", row="asymmetrical", hue="depth")
    # g.map(sns.lineplot, "resolution", "Accuracy")
    # g.add_legend()
    plt.savefig(os.path.join(result_path, "architecture_impact_depth.png"))
    plt.close()


def plot_dropout_impact(data_frame, result_path):
    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2], "scale": 0, "resolution": [16],
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

    properties = {
        "type": ["ae"],
        "clf": ["MLP"],
        "n_labels": 10000,
        "depth": [2], "scale": 0, "resolution": [16],
        "embedding_size": [256],
        "embedding_noise": 0.0,
        "task": "reconstruction",
        "task_difficulty": 0.0,
        "embedding_type": "glob_avg",
        "embedding_activation": "linear",
        "backbone": ["patch-residual"],
        "asymmetrical": False, "skip": False
    }
    lr_data_frame = select_properties(data_frame, properties)
    sns.lineplot(
        data=lr_data_frame,
        x="drop_rate",
        y="Accuracy",
        hue="dropout_structure",
        markers=True,
        style="dropout_structure"
    )
    plt.savefig(os.path.join(result_path, "dropout_impact_patch.png"))
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


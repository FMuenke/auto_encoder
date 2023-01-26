import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA

from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.barlow_twins.barlow_twin_network import BarlowTwinNetwork
from auto_encoder.sim_siam.sim_siam_network import SimpleSiameseNetwork
from auto_encoder.sim_clr.sim_clr_network import SimpleContrastiveLearning
from auto_encoder.nn_clr.nn_clr_network import NearestNeighbourCLRNetwork
from auto_encoder.vae.variational_auto_encoder import VariationalAutoEncoder

from auto_encoder.util import load_dict
from utils.ssl_lib import eval_semi_supervised_classification

from tqdm import tqdm


class Config:
    def __init__(self):
        self.opt = {}


def load_data_set(model, path_to_data, class_mapping):
    ds_test = DataSet(path_to_data)
    ds_test.load()
    images = ds_test.get_data()

    data_x = None
    data_y = []
    for i in tqdm(images):
        cls = i.load_y()
        if cls not in class_mapping:
            continue
        data_y.append(class_mapping[cls])
        data = i.load_x()
        pred = model.encode(data)
        if data_x is None:
            data_x = pred
        else:
            data_x = np.concatenate([data_x, pred], axis=0)
    data_y = np.array(data_y)
    return data_x, data_y


def get_data_sets(ds_path_train, ds_path_test, model_path, class_mapping):
    cfg = Config()
    if os.path.isfile(os.path.join(model_path, "opt.json")):
        cfg.opt = load_dict(os.path.join(model_path, "opt.json"))

    if "type" in cfg.opt:
        if cfg.opt["type"] == "variational-autoencoder":
            ae = VariationalAutoEncoder(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        elif cfg.opt["type"] == "autoencoder":
            ae = AutoEncoder(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        elif cfg.opt["type"] == "simsiam":
            ae = SimpleSiameseNetwork(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        elif cfg.opt["type"] == "barlowtwins":
            ae = BarlowTwinNetwork(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        elif cfg.opt["type"] == "simclr":
            ae = SimpleContrastiveLearning(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        elif cfg.opt["type"] == "nnclr":
            ae = NearestNeighbourCLRNetwork(model_path, cfg)
            ae.build(compile_model=False, add_decoder=False)
        else:
            raise Exception("UNKNOWN TYPE: {}".format(cfg.opt["type"]))
    else:
        ae = AutoEncoder(model_path, cfg)
        ae.build(add_decoder=False)

    x_train, y_train = load_data_set(ae, ds_path_train, class_mapping)
    x_test, y_test = load_data_set(ae, ds_path_test, class_mapping)
    return x_train, y_train, x_test, y_test


def vis_embeddings(x_train, y_train, x_test, y_test, path):
    n_components = 2
    print("Reducing to {} Components".format(n_components))
    if x_train.shape[1] > 2:
        decomp = PCA(n_components=n_components)
        x_train_dec = decomp.fit_transform(x_train)
        x_test_dec = decomp.transform(x_test)
    else:
        x_train_dec = x_train
        x_test_dec = x_test

    df_trn = pd.DataFrame({"x1": x_train_dec[:, 0], "x2": x_train_dec[:, 1], "class": y_train})
    df_tst = pd.DataFrame({"x1": x_test_dec[:, 0], "x2": x_test_dec[:, 1], "class": y_test})
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(ax=ax[0], data=df_trn, x="x1", y="x2", hue="class", style="class", legend=False)
    sns.scatterplot(ax=ax[1], data=df_tst, x="x1", y="x2", hue="class", style="class", legend=False)
    ax[0].set_title("Train")
    ax[1].set_title("Test")
    plt.savefig(os.path.join(path, "feature_space.png"))
    plt.close()

    df_trn = df_trn.groupby("class").mean()
    df_tst = df_tst.groupby("class").mean()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(ax=ax[0], data=df_trn, x="x1", y="x2", hue="class", style="class", legend=False)
    sns.scatterplot(ax=ax[1], data=df_tst, x="x1", y="x2", hue="class", style="class", legend=False)
    ax[0].set_title("Train")
    ax[1].set_title("Test")
    plt.savefig(os.path.join(path, "feature_space_class.png"))
    plt.close()


def main(args_):
    df = args_.dataset_folder
    model_path = args_.model
    direct_features = bool(args_.direct_features)

    class_mapping = load_dict(os.path.join(df, "class_mapping.json"))

    x_train, y_train, x_test, y_test = get_data_sets(
        os.path.join(df, "train"),
        os.path.join(df, "test"),
        model_path, class_mapping
    )

    # x_train = np.load(os.path.join(model_path, "x_train{}.npy".format(ds_ident)))
    # y_train = np.load(os.path.join(model_path, "y_train{}.npy".format(ds_ident)))

    # x_test = np.load(os.path.join(model_path, "x_test{}.npy".format(ds_ident)))
    # y_test = np.load(os.path.join(model_path, "y_test{}.npy".format(ds_ident)))

    vis_embeddings(x_train, y_train, x_test, y_test, model_path)

    eval_semi_supervised_classification(x_train, y_train, x_test, y_test, model_path, direct_features=direct_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", "-df", help="Path to directory with dataset")
    parser.add_argument("--model", "-m", help="Path to model")
    parser.add_argument("--direct_features", "-direct", default=False, help="Use features directly from feature-map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

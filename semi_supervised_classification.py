import os
import argparse
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.variational_auto_encoder import VariationalAutoEncoder

from auto_encoder.util import save_dict, load_dict
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
            ae.build(add_decoder=False)
        elif cfg.opt["type"] == "autoencoder":
            ae = AutoEncoder(model_path, cfg)
            ae.build(add_decoder=False)
        else:
            raise Exception("UNKNOWN TYPE: {}".format(cfg.opt["type"]))
    else:
        ae = AutoEncoder(model_path, cfg)
        ae.build(add_decoder=False)

    x_train, y_train = load_data_set(ae, ds_path_train, class_mapping)
    np.save(os.path.join(model_path, "x_train.npy"), x_train)
    np.save(os.path.join(model_path, "y_train.npy"), y_train)

    x_test, y_test = load_data_set(ae, ds_path_test, class_mapping)
    np.save(os.path.join(model_path, "x_test.npy"), x_test)
    np.save(os.path.join(model_path, "y_test.npy"), y_test)


def main(args_):
    df = args_.dataset_folder
    model_path = args_.model

    class_mapping = load_dict(os.path.join(df, "class_mapping.json"))

    if not os.path.isfile(os.path.join(model_path, "y_test.npy")):
        get_data_sets(os.path.join(df, "train"), os.path.join(df, "test"), model_path, class_mapping)

    x_train = np.load(os.path.join(model_path, "x_train.npy"))
    y_train = np.load(os.path.join(model_path, "y_train.npy"))

    x_test = np.load(os.path.join(model_path, "x_test.npy"))
    y_test = np.load(os.path.join(model_path, "y_test.npy"))

    eval_semi_supervised_classification(x_train, y_train, x_test, y_test, model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with dataset",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Path to model",
    )
    parser.add_argument(
        "--testset_folder",
        "-tf",
        help="Path to directory with dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

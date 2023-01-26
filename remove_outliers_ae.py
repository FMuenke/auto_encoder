import cv2
import os
import argparse
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.vae.variational_auto_encoder import VariationalAutoEncoder
from auto_encoder.sim_clr.sim_clr_network import SimpleContrastiveLearning
from auto_encoder.sim_siam.sim_siam_network import SimpleSiameseNetwork
from auto_encoder.nn_clr.nn_clr_network import NearestNeighbourCLRNetwork
from auto_encoder.barlow_twins.barlow_twin_network import BarlowTwinNetwork

from auto_encoder.util import load_dict

import pandas as pd
from utils.outlier_removal import eval_outlier_removal

from tqdm import tqdm


class Config:
    def __init__(self):
        self.opt = {}


def make_result_picture(img, res):
    res = res * 255
    img = cv2.resize(img, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)
    block = np.ones((img.shape[0], 10, 3))
    complete = np.concatenate([img, block, res], axis=1)
    return complete


def load_data_set(model, path_to_data, known_classes, only_known_classes):
    ds_test = DataSet(path_to_data)
    ds_test.load()
    images = ds_test.get_data()

    class_mapping = {cls: i+1 for i, cls in enumerate(known_classes)}

    data_x = None
    data_y = []
    data_frame = {"class_name": [], "status": [], "status_id": []}
    for i in tqdm(images):
        cls = i.load_y()
        if cls not in known_classes and only_known_classes:
            continue

        if cls not in known_classes:
            data_frame["status"].append("unknown")
            data_frame["status_id"].append(0)
            data_y.append(0)
        else:
            data_frame["status"].append("known")
            data_frame["status_id"].append(1)
            data_y.append(class_mapping[cls])
        data = i.load_x()
        pred = model.encode(data)

        if data_x is None:
            data_x = pred
        else:
            data_x = np.concatenate([data_x, pred], axis=0)

        data_frame["class_name"].append(cls)
    data_frame["cls_id"] = data_y
    data_y = np.array(data_y)
    return data_x, data_y, data_frame


def get_data_sets(ds_path, model_path):
    ds_path_train = os.path.join(ds_path, "train")
    ds_path_test = os.path.join(ds_path, "test")

    class_mapping = load_dict(os.path.join(ds_path, "class_mapping.json"))

    cfg = Config()
    if os.path.isfile(os.path.join(model_path, "opt.json")):
        cfg.opt = load_dict(os.path.join(model_path, "opt.json"))

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

    x_train, y_train, data_frame_train = load_data_set(ae, ds_path_train, class_mapping, only_known_classes=True)
    x_test, y_test, data_frame_test = load_data_set(ae, ds_path_test, class_mapping, only_known_classes=False)
    return x_train, y_train, data_frame_train, x_test, y_test, data_frame_test


def main(args_):
    df = args_.dataset_folder
    model_path = args_.model

    x_train, y_train, data_frame_train, x_test, y_test, data_frame_test = get_data_sets(df, model_path)

    data_frame_train = pd.DataFrame(data_frame_train)
    data_frame_test = pd.DataFrame(data_frame_test)

    eval_outlier_removal(x_train, y_train, x_test, y_test, data_frame_test, model_path)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

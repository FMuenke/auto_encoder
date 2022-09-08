import cv2
import os
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.util import check_n_make_dir, load_dict

import argparse


class Config:
    def __init__(self):
        self.opt = {
            "backbone": "basic",
            "loss": "mse",
            "optimizer": "lazy_adam",
            "epochs": 10000,
            "batch_size": 16,
            "embedding_size": 512,
            "init_learning_rate": 1e-5,
            "input_shape": [256, 256, 3],
        }


def make_result_picture(img, res):
    res = np.clip((res + 0.5) * 255, 0, 255)
    img = cv2.resize(img, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)
    block = np.ones((img.shape[0], 10, 3))
    complete = np.concatenate([img, block, res], axis=1)
    return complete


def main(args_):
    mf = args_.model
    df = args_.dataset_folder

    ds = DataSet(df)
    ds.load()
    test_images = ds.get_data()

    cfg = Config()
    if os.path.isfile(os.path.join(mf, "opt.json")):
        cfg.opt = load_dict(os.path.join(mf, "opt.json"))

    results_folder = os.path.join(mf, "results")
    check_n_make_dir(results_folder, True)

    ae = AutoEncoder(mf, cfg)
    ae.build(False)

    for i in test_images:
        print(i.name)
        data = i.load_x()
        pred = ae.inference(data)
        pred = pred[0, :, :, :]

        cv2.imwrite(os.path.join(results_folder, i.name[:-4] + ".png"), make_result_picture(data, pred))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="./test/",
        help="Path to model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


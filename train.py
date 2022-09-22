import os

from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.augmentations import Augmentations

from auto_encoder.util import save_dict, check_n_make_dir

import argparse


class Config:
    def __init__(self):
        self.opt = {
            "backbone": "fully_connected",
            "resolution": 4,
            "depth": 4,
            "loss": "mse",
            "optimizer": "adam",
            "batch_size": 64,
            "embedding_size": 256,
            "init_learning_rate": 1e-3,
            "input_shape": [128, 128, 3],
        }


def main(args_):

    mf = args_.model
    df = args_.dataset_folder
    ds = DataSet(df)

    ae = AutoEncoder(mf, Config())
    ae.build(add_decoder=True)

    ds.load()
    train_images, test_image = ds.get_data(0.8)

    augmentations = Augmentations(color=False)

    check_n_make_dir(mf)
    save_dict(Config().opt, os.path.join(mf, "opt.json"))
    ae.fit(train_images, test_image, augmentations)


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


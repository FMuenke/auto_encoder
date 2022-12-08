import os
import tensorflow as tf
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.variational_auto_encoder import VariationalAutoEncoder

from auto_encoder.augmentations import Augmentations, EncoderTask

from auto_encoder.util import save_dict, check_n_make_dir

import argparse

print("TF VERSION: ", tf.__version__)


class Config:
    def __init__(self):
        self.opt = {
            "type": "autoencoder",
            "task": "reconstruction",  # reconstruction, denoise, completion
            "task_difficulty": 0.25,
            "backbone": "residual",
            "resolution": 16,
            "depth": 2,
            "optimizer": "adam",
            "batch_size": 128,
            "embedding_size": 128,
            "embedding_type": "glob_avg",
            "embedding_activation": "linear",
            "drop_rate": 0.0,
            "init_learning_rate": 1e-3,
            "input_shape": [32, 32, 3],
            "tf-version": tf.__version__,
        }


def main(args_):

    mf = args_.model
    df = args_.dataset_folder
    ds = DataSet(df)

    cfg = Config()
    cfg.opt["type"] = args_.type
    cfg.opt["task"] = args_.task
    cfg.opt["task_difficulty"] = float(args_.task_difficulty)
    cfg.opt["embedding_size"] = int(args_.embedding_size)
    cfg.opt["embedding_type"] = args_.embedding_type
    cfg.opt["embedding_activation"] = args_.embedding_activation
    cfg.opt["drop_rate"] = float(args_.drop_rate)
    cfg.opt["embedding_noise"] = float(args_.embedding_noise)

    cfg.opt["backbone"] = args_.backbone
    cfg.opt["resolution"] = int(args_.resolution)
    cfg.opt["depth"] = int(args_.depth)
    cfg.opt["skip"] = args_.use_skip
    cfg.opt["asymmetrical"] = args_.use_asymmetrical

    if "type" in cfg.opt:
        if cfg.opt["type"] == "variational-autoencoder":
            ae = VariationalAutoEncoder(mf, cfg)
            ae.build(add_decoder=True)
        elif cfg.opt["type"] == "autoencoder":
            ae = AutoEncoder(mf, cfg)
            ae.build(add_decoder=True)
        else:
            raise Exception("UNKNOWN TYPE: {}".format(cfg.opt["type"]))
    else:
        ae = AutoEncoder(mf, cfg)
        ae.build(add_decoder=True)

    ds.load()
    train_images, test_image = ds.get_data(0.8)

    if cfg.opt["task"] == "reconstruction":
        task = None
    elif cfg.opt["task"] == "blurring":
        task = EncoderTask(blurring=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "denoise":
        task = EncoderTask(noise=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "completion_cross_cut":
        task = EncoderTask(cross_cut=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "completion_masking":
        task = EncoderTask(masking=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "reconstruction_shuffled":
        task = EncoderTask(patch_shuffling=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "reconstruction_rotated":
        task = EncoderTask(patch_rotation=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "completion_blackhole":
        task = EncoderTask(black_hole=cfg.opt["task_difficulty"])
    else:
        raise Exception("No Valid Task Specified.. {}".format(cfg.opt["task"]))

    check_n_make_dir(mf)
    save_dict(cfg.opt, os.path.join(mf, "opt.json"))
    ae.fit(train_images, test_image, task)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    parser.add_argument("--model", "-m", help="Path to model")
    parser.add_argument("--type", "-ty", default="autoencoder", help="Path to model")
    parser.add_argument("--task", "-t", default="reconstruction", help="Path to model")
    parser.add_argument("--task_difficulty", "-difficulty", default=0.0, help="Training Mode")
    parser.add_argument("--embedding_size", "-size", default=128, help="Training Mode")
    parser.add_argument("--embedding_type", "-type", default="glob_avg", help="Training Mode")
    parser.add_argument("--embedding_activation", "-activation", default="leaky_relu", help="Training Mode")
    parser.add_argument("--drop_rate", "-drop", default=0.0, help="Dropout during Embedding")
    parser.add_argument("--embedding_noise", "-noise", default=0.0, help="Gaussian Noise applied to embedding")
    parser.add_argument("--backbone", "-bb", default="residual", help="Auto Encoder Backbone")
    parser.add_argument("--depth", "-d", default=4, help="Backbone Depth")
    parser.add_argument("--resolution", "-r", default=4, help="Backbone Resolution")
    parser.add_argument("--use_skip", "-s", default=False, type=bool, help="Add a skip connection to the bottleneck")
    parser.add_argument("--use_asymmetrical", "-asym", default=False, type=bool, help="Reduce the decoder size to a minimum")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


import os
import tensorflow as tf
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.simple_siamse_network import SimpleSiameseNetwork
from auto_encoder.barlow_twin_network import BarlowTwinNetwork
from auto_encoder.variational_auto_encoder import VariationalAutoEncoder

from auto_encoder.augmentations import Augmentations, EncoderTask

from auto_encoder.util import save_dict, check_n_make_dir

import argparse

print("TF VERSION: ", tf.__version__)


class Config:
    def __init__(self):
        self.opt = {
            "optimizer": "adam",
            "batch_size": 128,
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
    cfg.opt["init_learning_rate"] = float(args_.learning_rate)
    cfg.opt["batch_size"] = int(args_.batch_size)
    cfg.opt["input_shape"] = [int(args_.input_size), int(args_.input_size), 3]

    cfg.opt["task_difficulty"] = float(args_.task_difficulty)
    cfg.opt["embedding_size"] = int(args_.embedding_size)
    cfg.opt["embedding_type"] = args_.embedding_type
    cfg.opt["embedding_activation"] = args_.embedding_activation
    cfg.opt["drop_rate"] = float(args_.drop_rate)
    cfg.opt["dropout_structure"] = args_.dropout_structure

    cfg.opt["backbone"] = args_.backbone
    cfg.opt["resolution"] = int(args_.resolution)
    cfg.opt["depth"] = int(args_.depth)
    cfg.opt["scale"] = int(args_.scale)
    cfg.opt["asymmetrical"] = args_.use_asymmetrical

    if cfg.opt["type"] == "variational-autoencoder":
        ae = VariationalAutoEncoder(mf, cfg)
        ae.build(add_decoder=True)
    elif cfg.opt["type"] == "autoencoder":
        ae = AutoEncoder(mf, cfg)
        ae.build(add_decoder=True)
    elif cfg.opt["type"] == "simsiam":
        ae = SimpleSiameseNetwork(mf, cfg)
        ae.build(True)
    elif cfg.opt["type"] == "barlowtwin":
        ae = BarlowTwinNetwork(mf, cfg)
        ae.build(True)
    else:
        raise Exception("UNKNOWN TYPE: {}".format(cfg.opt["type"]))

    ds.load()
    train_images, test_image = ds.get_data(0.8)

    if cfg.opt["task"] == "baseline":
        task = None
    elif cfg.opt["task"] == "blurring":
        task = EncoderTask(blurring=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "noise":
        task = EncoderTask(noise=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "cross_cut":
        task = EncoderTask(cross_cut=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "masking":
        task = EncoderTask(masking=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "patches_shuffled":
        task = EncoderTask(patch_shuffling=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "patches_rotated":
        task = EncoderTask(patch_rotation=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "warp":
        task = EncoderTask(warp=cfg.opt["task_difficulty"])
    elif cfg.opt["task"] == "patches_masked":
        task = EncoderTask(imagine_patches=cfg.opt["task_difficulty"])
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
    parser.add_argument("--learning_rate", "-lr", default=1e-3, help="Path to model")
    parser.add_argument("--batch_size", "-batch", default=128, help="Path to model")
    parser.add_argument("--input_size", "-in", default=32, help="Path to model")
    parser.add_argument("--type", "-ty", default="autoencoder", help="Path to model")
    parser.add_argument("--task", "-t", default="baseline", help="Task for the autoencoder")
    parser.add_argument("--task_difficulty", "-dif", default=0.0, help="Difficulty of Task [%]")
    parser.add_argument("--embedding_size", "-esize", default=256, help="Size of Embedding/Latent Space")
    parser.add_argument("--embedding_type", "-etype", default="glob_avg", help="Type of Embedding [Flatten/GlobalAverage/..]")
    parser.add_argument("--embedding_activation", "-eact", default="linear", help="Activation Funktion of Embeding")
    parser.add_argument("--drop_rate", "-drop", default=0.0, help="Dropout for the Embedding")
    parser.add_argument("--dropout_structure", "-drops", default="general", help="Dropout Type for Embedding")
    parser.add_argument("--backbone", "-bb", default="residual", help="Auto Encoder Backbone")
    parser.add_argument("--depth", "-d", default=2, help="Backbone Depth")
    parser.add_argument("--scale", "-s", default=0, help="Backbone Scale")
    parser.add_argument("--resolution", "-r", default=16, help="Backbone Resolution")
    parser.add_argument("--use_asymmetrical", "-asym", default=False, type=bool, help="Reduce the decoder size to a minimum")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


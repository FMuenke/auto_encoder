import os
import tensorflow as tf
from auto_encoder.data_set import DataSet, sample_images_by_class
from auto_encoder.image_classifier import ImageClassifier

from auto_encoder.augmentations import Augmentations, EncoderTask

from auto_encoder.util import save_dict, check_n_make_dir, load_dict

import argparse

print("TF VERSION: ", tf.__version__)


class Config:
    def __init__(self):
        self.opt = {
            "optimizer": "adam",
            "batch_size": 128,
            "init_learning_rate": 1e-3,
            "input_shape": [32, 32, 3],
            "tf-version": tf.__version__,
        }


def main(args_):

    mf = args_.model
    df = args_.dataset_folder
    ds = DataSet(os.path.join(df, "train"))

    cfg = Config()
    cfg.opt["n_labels"] = int(args_.n_labels)
    cfg.opt["type"] = args_.type
    cfg.opt["task"] = args_.task
    cfg.opt["task_difficulty"] = float(args_.task_difficulty)
    cfg.opt["embedding_size"] = int(args_.embedding_size)
    cfg.opt["embedding_type"] = args_.embedding_type
    cfg.opt["embedding_activation"] = args_.embedding_activation
    cfg.opt["drop_rate"] = float(args_.drop_rate)
    cfg.opt["dropout_structure"] = args_.dropout_structure
    cfg.opt["embedding_noise"] = float(args_.embedding_noise)

    cfg.opt["backbone"] = args_.backbone
    cfg.opt["resolution"] = int(args_.resolution)
    cfg.opt["depth"] = int(args_.depth)

    class_mapping = load_dict(os.path.join(df, "class_mapping.json"))

    clf = ImageClassifier(mf, cfg, class_mapping)
    clf.build(True)

    ds.load()
    if cfg.opt["n_labels"] > 0:
        images = ds.get_data()
        train_images, remaining_images = sample_images_by_class(images, int(cfg.opt["n_labels"] * 0.8), class_mapping)
        test_images, _ = sample_images_by_class(remaining_images, int(cfg.opt["n_labels"] * 0.2), class_mapping)
    else:
        train_images, test_images = ds.get_data(0.8)

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
    clf.fit(train_images, test_images, task)


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
    parser.add_argument("--embedding_size", "-size", default=256, help="Training Mode")
    parser.add_argument("--embedding_type", "-type", default="glob_avg", help="Training Mode")
    parser.add_argument("--embedding_activation", "-activation", default="linear", help="Training Mode")
    parser.add_argument("--drop_rate", "-drop", default=0.0, help="Dropout during Embedding")
    parser.add_argument("--dropout_structure", "-drops", default="general", help="Dropout during Embedding")
    parser.add_argument("--embedding_noise", "-noise", default=0.0, help="Gaussian Noise applied to embedding")
    parser.add_argument("--backbone", "-bb", default="residual", help="Auto Encoder Backbone")
    parser.add_argument("--depth", "-d", default=2, help="Backbone Depth")
    parser.add_argument("--resolution", "-r", default=16, help="Backbone Resolution")
    parser.add_argument("--n_labels", "-n", default=0, help="Number of Samples to train with")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


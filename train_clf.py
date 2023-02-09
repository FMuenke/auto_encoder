import os
import shutil
import numpy as np
import tensorflow as tf
from auto_encoder.data_set import DataSet, sample_images_by_class
from auto_encoder.image_classifier import ImageClassifier
from auto_encoder.hybrid_image_classifier import HybridImageClassifier

from auto_encoder.augmentations import Augmentations

from auto_encoder.util import save_dict, check_n_make_dir, load_dict

import argparse

print("TF VERSION: ", tf.__version__)


class Config:
    def __init__(self):
        self.opt = {
            "optimizer": "adam",
            "batch_size": 128,
            # "init_learning_rate": 1e-3,
            "input_shape": [32, 32, 3],
            "tf-version": tf.__version__,
        }


def main(args_):

    mf = args_.model
    df = args_.dataset_folder
    ds = DataSet(os.path.join(df, "train"))

    pretrained_weights_path = args_.weights
    if os.path.isfile(pretrained_weights_path):
        check_n_make_dir(mf)
        shutil.copy(
            pretrained_weights_path,
            os.path.join(mf, "ae-weights-final.hdf5")
        )

    cfg = Config()
    cfg.opt["n_labels"] = int(args_.n_labels)
    cfg.opt["init_learning_rate"] = float(args_.learning_rate)
    cfg.opt["freeze"] = args_.freeze_backbone
    cfg.opt["input_shape"] = [int(args_.input_size), int(args_.input_size), 3]

    cfg.opt["augmentation"] = args_.augmentation
    cfg.opt["embedding_size"] = int(args_.embedding_size)
    cfg.opt["embedding_type"] = args_.embedding_type
    cfg.opt["embedding_activation"] = args_.embedding_activation
    cfg.opt["drop_rate"] = float(args_.drop_rate)
    cfg.opt["dropout_structure"] = args_.dropout_structure

    cfg.opt["type"] = args_.type
    cfg.opt["backbone"] = args_.backbone
    cfg.opt["resolution"] = int(args_.resolution)
    cfg.opt["depth"] = int(args_.depth)
    cfg.opt["scale"] = int(args_.scale)
    cfg.opt["asymmetrical"] = bool(args_.asymmetrical)
    cfg.opt["batch_size"] = int(args_.batch_size)

    class_mapping = load_dict(os.path.join(df, "class_mapping.json"))

    if cfg.opt["type"] == "hybrid":
        clf = HybridImageClassifier(mf, cfg, class_mapping)
    else:
        clf = ImageClassifier(mf, cfg, class_mapping)
    clf.build(True)

    ds.load()
    counts = ds.count(class_mapping)
    for c in counts:
        print("[INFO] {}: {}".format(c, counts[c]))
    if cfg.opt["n_labels"] > 0:
        images = ds.get_data()
        train_images, remaining_images = sample_images_by_class(images, int(cfg.opt["n_labels"] * 0.80), class_mapping)
        test_images, _ = sample_images_by_class(remaining_images, int(cfg.opt["n_labels"] * 0.10), class_mapping)
    else:
        images = ds.get_data()
        min_nb = np.min([counts[c] for c in counts])
        min_nb = min_nb * len(class_mapping)
        test_images, train_images = sample_images_by_class(images, int(min_nb * 0.10), class_mapping)

    if cfg.opt["augmentation"] == "None":
        aug = None
    elif cfg.opt["augmentation"] == "all":
        aug = Augmentations(
            neutral_percentage=0.85,
            blurring=0.1,
            masking=0.15,
            patch_shuffling=0.15,
            noise=0.20,
            channel_shift=0.05,
            brightness=0.10,
            crop=0.20,
            warp=0.05,
        )
    else:
        raise Exception("No Valid Task Specified.. {}".format(cfg.opt["task"]))

    check_n_make_dir(mf)
    save_dict(cfg.opt, os.path.join(mf, "opt.json"))
    clf.fit(train_images, test_images, aug)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    parser.add_argument("--model", "-m", help="Path to model")
    parser.add_argument("--input_size", "-in", default=32, help="Path to model")
    parser.add_argument("--type", "-ty", default="", help="Model Type [cnn / hybrid]")
    parser.add_argument("--asymmetrical", "-asym", default=False, type=bool)
    parser.add_argument("--weights", "-w", default="None", help="Path to pretrained weights")
    parser.add_argument("--freeze_backbone", "-freeze", type=bool, default=False, help="Path to pretrained weights")
    parser.add_argument("--augmentation", "-aug", default="None", help="Path to model")
    parser.add_argument("--embedding_size", "-size", default=256, help="Training Mode")
    parser.add_argument("--embedding_type", "-type", default="glob_avg", help="Training Mode")
    parser.add_argument("--embedding_activation", "-activation", default="linear", help="Training Mode")
    parser.add_argument("--drop_rate", "-drop", default=0.0, help="Dropout during Embedding")
    parser.add_argument("--dropout_structure", "-drops", default="general", help="Dropout during Embedding")
    parser.add_argument("--backbone", "-bb", default="residual", help="Auto Encoder Backbone")
    parser.add_argument("--depth", "-d", default=2, help="Backbone Depth")
    parser.add_argument("--resolution", "-r", default=16, help="Backbone Resolution")
    parser.add_argument("--scale", "-s", default=0, help="Backbone Scale")
    parser.add_argument("--n_labels", "-n", default=0, help="Number of Samples to train with")
    parser.add_argument("--learning_rate", "-lr", default=0.001, help="Initial Learning Rate")
    parser.add_argument("--batch_size", "-batch", default=128, help="Batch Size used for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


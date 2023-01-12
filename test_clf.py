import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from auto_encoder.data_set import DataSet
from auto_encoder.image_classifier import ImageClassifier
from auto_encoder.hybrid_image_classifier import HybridImageClassifier

from auto_encoder.util import save_dict, check_n_make_dir, load_dict

import argparse

from sklearn.metrics import classification_report, f1_score, accuracy_score

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
    ds = DataSet(os.path.join(df, "test"))

    class_mapping = load_dict(os.path.join(df, "class_mapping.json"))
    class_mapping_inv = {v: k for k, v in class_mapping.items()}

    cfg = Config()
    if os.path.isfile(os.path.join(mf, "opt.json")):
        cfg.opt = load_dict(os.path.join(mf, "opt.json"))
    else:
        raise Exception("NO CONFIG FOUND!")

    if cfg.opt["type"] == "hybrid":
        clf = HybridImageClassifier(mf, cfg, class_mapping)
    else:
        clf = ImageClassifier(mf, cfg, class_mapping)

    clf.build(False)

    ds.load()
    images = ds.get_data()

    y_true = []
    y_pred = []

    for i in tqdm(images):
        data = i.load_x()
        yi_true = i.load_y()
        yi_pred = class_mapping_inv[clf.inference(data)]
        y_true.append(yi_true)
        y_pred.append(yi_pred)

    with open(os.path.join(mf, "clf-report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))

    results_df = pd.DataFrame([{
        "clf": "cnn",
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "run": 0,
        "n_labels": cfg.opt["n_labels"]
    }])
    results_df.to_csv(os.path.join(mf, "classifier_results.csv"))

    print(results_df)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    parser.add_argument("--model", "-m", help="Path to model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


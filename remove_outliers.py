import cv2
import os
import argparse
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.util import check_n_make_dir, save_dict, load_dict

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import LocalOutlierFactor
from umap import UMAP

from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

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
        pred = model.inference(data)

        if data_x is None:
            data_x = pred
        else:
            data_x = np.concatenate([data_x, pred], axis=0)

        data_frame["class_name"].append(cls)
    data_frame["cls_id"] = data_y
    data_y = np.array(data_y)
    return data_x, data_y, data_frame


def get_data_sets(ds_path_train, ds_path_test, model_path):
    cfg = Config()
    if os.path.isfile(os.path.join(model_path, "opt.json")):
        cfg.opt = load_dict(os.path.join(model_path, "opt.json"))
    ae = AutoEncoder(model_path, cfg)
    ae.build(False, add_decoder=False)

    known_classes = ["manhole", "stormdrain"]

    x_train, y_train, data_frame_train = load_data_set(ae, ds_path_train, known_classes, only_known_classes=True)
    np.save(os.path.join(model_path, "x_train.npy"), x_train)
    np.save(os.path.join(model_path, "y_train.npy"), y_train)
    save_dict(data_frame_train, os.path.join(model_path, "data_frame_train.json"))

    x_test, y_test, data_frame_test = load_data_set(ae, ds_path_test, known_classes, only_known_classes=False)
    np.save(os.path.join(model_path, "x_test.npy"), x_test)
    np.save(os.path.join(model_path, "y_test.npy"), y_test)
    save_dict(data_frame_test, os.path.join(model_path, "data_frame_test.json"))


def main(args_):
    df = args_.dataset_folder
    tf = args_.testset_folder
    model_path = args_.model

    get_data_sets(df, tf, model_path)

    x_train = np.load(os.path.join(model_path, "x_train.npy"))
    y_train = np.load(os.path.join(model_path, "y_train.npy"))
    data_frame_train = load_dict(os.path.join(model_path, "data_frame_train.json"))

    x_test = np.load(os.path.join(model_path, "x_test.npy"))
    y_test = np.load(os.path.join(model_path, "y_test.npy"))
    data_frame_test = load_dict(os.path.join(model_path, "data_frame_test.json"))

    data_frame_train = pd.DataFrame(data_frame_train)
    data_frame_test = pd.DataFrame(data_frame_test)

    s = ""

    s += "[INFO] CLASSIFICATION\n"
    rf_classifier = RandomForestClassifier(n_jobs=-1)
    rf_classifier.fit(x_train, y_train)
    y_cls_only = rf_classifier.predict(x_test[y_test != 0, :])
    s += "[RANDOM FORREST CLASSIFIER - F1-SCORE] {}\n".format(f1_score(y_test[y_test != 0], y_cls_only))

    proba = rf_classifier.predict_log_proba(x_test)
    max_proba = np.max(proba, axis=1)
    s += "[RANDOM FORREST CLASSIFIER - AUROC] {}\n".format(roc_auc_score(data_frame_test["status_id"], max_proba))

    s += "\n[INFO] Fitting outlier removal...\n"
    iso_remover = IsolationForest(n_jobs=-1)
    lof_remover = LocalOutlierFactor(n_jobs=-1, novelty=True)
    iso_remover.fit(x_train)
    lof_remover.fit(x_train)

    s += "OUTLIER CLASSIFICATION REPORT\n"
    outlier_score = iso_remover.score_samples(x_test)
    s += "[ISOLATION FORREST] {}\n".format(roc_auc_score(data_frame_test["status_id"], outlier_score))
    outlier_score = lof_remover.score_samples(x_test)
    s += "[LOCAL OUTLIER FACTOR] {}\n".format(roc_auc_score(data_frame_test["status_id"], outlier_score))

    print(s)
    with open(os.path.join(model_path, "outlier-results.txt"), "w") as f:
        f.write(s)

    print("[INFO] UMAP")
    projection = UMAP(n_components=4)
    projection.fit(x_train)

    x_trans_test = projection.transform(x_test)
    plt_df = pd.DataFrame({
        "x1": x_trans_test[:, 0],
        "x2": x_trans_test[:, 1],
        "x3": x_trans_test[:, 2],
        "x4": x_trans_test[:, 3],
        "status": data_frame_test["status"],
        "class_name": data_frame_test["class_name"],
    })

    sns.pairplot(data=plt_df, vars=["x1", "x2", "x3", "x4"], hue="class_name", kind="kde")
    plt.savefig(os.path.join(model_path, "umap-dist.png"))
    plt.show()


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

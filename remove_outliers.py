import cv2
import os
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.util import check_n_make_dir, save_dict, load_dict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from umap import UMAP

from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


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
    res = res * 255
    img = cv2.resize(img, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)
    block = np.ones((img.shape[0], 10, 3))
    complete = np.concatenate([img, block, res], axis=1)
    return complete


def load_data_set(model, path_to_data, class_mapping=None, cls_to_consider=None):
    ds_test = DataSet(path_to_data)
    ds_test.load()
    images = ds_test.get_data()

    data_x = None
    data_y = []
    data_frame = {"class_name": [], "cls_id": []}
    if class_mapping is None:
        class_mapping = {}

    for i in tqdm(images):
        cls = i.load_y()
        if cls_to_consider is not None:
            if cls not in cls_to_consider:
                continue
        data = i.load_x()
        if cls not in class_mapping:
            class_mapping[cls] = len(class_mapping)
        pred = model.inference(data)

        if data_x is None:
            data_x = pred
        else:
            data_x = np.concatenate([data_x, pred], axis=0)
        data_y.append(class_mapping[cls])

        data_frame["class_name"].append(cls)
        data_frame["cls_id"].append(class_mapping[cls])
    data_y = np.array(data_y)
    return data_x, data_y, data_frame, class_mapping


def get_data_sets(ds_path, model_path):
    cfg = Config()
    if os.path.isfile(os.path.join(model_path, "opt.json")):
        cfg.opt = load_dict(os.path.join(model_path, "opt.json"))
    ae = AutoEncoder(model_path, cfg)
    ae.build(False, add_decoder=False)

    x_train, y_train, data_frame_train, class_mapping = load_data_set(ae, os.path.join(ds_path, "train", "known"))
    np.save(os.path.join(model_path, "x_train.npy"), x_train)
    np.save(os.path.join(model_path, "y_train.npy"), y_train)
    save_dict(data_frame_train, os.path.join(model_path, "data_frame_train.json"))
    save_dict(class_mapping, os.path.join(model_path, "class_mapping_train.json"))

    x_test, y_test, data_frame_test, class_mapping = load_data_set(ae, os.path.join(ds_path, "test"), class_mapping)
    np.save(os.path.join(model_path, "x_test.npy"), x_test)
    np.save(os.path.join(model_path, "y_test.npy"), y_test)
    save_dict(data_frame_test, os.path.join(model_path, "data_frame_test.json"))
    save_dict(class_mapping, os.path.join(model_path, "class_mapping_test.json"))


def mark_unknowns(data_frame, class_mapping):
    known_cls = []
    known_id = []

    for c in data_frame["class_name"]:
        if c in class_mapping:
            known_cls.append("known")
            known_id.append(1)
        else:
            known_cls.append("unknown")
            known_id.append(0)
    return known_cls, np.array(known_id)


def remove_outliers(x_train, y_train, x_test, y_test, class_mapping_train, data_frame_test):
    print("[INFO] Fitting outlier removal...")
    outlier_remover = IsolationForest(n_estimators=1000, n_jobs=-1)
    nn_remover = LocalOutlierFactor(n_jobs=-1, novelty=True)
    outlier_remover.fit(x_train)
    nn_remover.fit(x_train)

    unknown_cls, unknown_id = mark_unknowns(data_frame_test, class_mapping_train)

    data_frame_test["unknown"] = unknown_cls

    print("OUTLIER CLASSIFICATION REPORT")
    outlier_score = outlier_remover.score_samples(x_test)
    auroc = roc_auc_score(unknown_id, outlier_score)
    print("[ISOLATION FORREST] ", 1 - auroc)
    outlier_score = nn_remover.score_samples(x_test)
    auroc = roc_auc_score(unknown_id, outlier_score)
    print("[NN] ", auroc)


def classify_knows(x_train, y_train, x_test, y_test, unknown_cls):
    print("[INFO] Fitting classifier...")
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    x_test_known = x_test[unknown_cls == "known", :]
    y_test_known = y_test[unknown_cls == "known"]
    y_pred = classifier.predict(x_test_known)
    print("CLASSIFIER CLASSIFICATION REPORT [KNOWN SAMPLES] - F1-Score: {}".format(
        f1_score(y_test_known, y_pred, average="weighted")))
    print(classification_report(y_test_known, y_pred))


def apply_umap():
    dim_reducer = UMAP(n_components=2)
    dim_reducer.fit(x_train)
    x_t = dim_reducer.transform(x_train)
    data_frame_train["x1"] = x_t[:, 0]
    data_frame_train["x2"] = x_t[:, 1]

    x_t = dim_reducer.transform(x_test)
    data_frame_test["x1"] = x_t[:, 0]
    data_frame_test["x2"] = x_t[:, 1]
    data_frame_test["class_name"] = data_frame_test["class_name"]
    data_frame_test["cls_id"] = y_test
    data_frame_test = pd.DataFrame(data_frame_test)
    fig, axes = plt.subplots(1, 2)
    sns.displot(data_frame_test, x="x1", y="x2", hue="status", legend=True, kind="kde", ax=axes[0])
    sns.displot(data_frame_test, x="x1", y="x2", hue="class_name", legend=False, kind="kde", ax=axes[1])
    plt.show()


def main():
    ds_path = "/home/fmuenke/datasets/TS-DATA-GROUPED"
    model_path = "/home/fmuenke/ai_models/AE-RES-ENC64-D4-R1"

    get_data_sets(ds_path, model_path)

    x_train = np.load(os.path.join(model_path, "x_train.npy"))
    y_train = np.load(os.path.join(model_path, "y_train.npy"))
    data_frame_train = load_dict(os.path.join(model_path, "data_frame_train.json"))
    class_mapping_train = load_dict(os.path.join(model_path, "class_mapping_train.json"))

    x_test = np.load(os.path.join(model_path, "x_test.npy"))
    y_test = np.load(os.path.join(model_path, "y_test.npy"))
    data_frame_test = load_dict(os.path.join(model_path, "data_frame_test.json"))

    unknown_cls_train, unknown_id_train = mark_unknowns(data_frame_train, class_mapping_train)
    unknown_cls, unknown_id = mark_unknowns(data_frame_test, class_mapping_train)
    data_frame_test["status"] = unknown_cls

    remove_outliers(x_train, y_train, x_test, y_test, class_mapping_train, data_frame_test)
    # classify_knows(x_train, y_train, x_test, y_test, unknown_cls)

    dim_reducer = UMAP(n_components=4)
    dim_reducer.fit(x_train)
    x_t = dim_reducer.transform(x_train)
    data_frame_train["x1"] = x_t[:, 0]
    data_frame_train["x2"] = x_t[:, 1]

    x_t = dim_reducer.transform(x_test)
    data_frame_test["x1"] = x_t[:, 0]
    data_frame_test["x2"] = x_t[:, 1]
    data_frame_test["x3"] = x_t[:, 2]
    data_frame_test["x4"] = x_t[:, 3]
    data_frame_test["class_name"] = data_frame_test["class_name"]
    data_frame_test["cls_id"] = y_test
    data_frame_test = pd.DataFrame(data_frame_test)
    ### PAIRPLOTS?!?!?!
    # sns.displot(data=data_frame_test, x="x1", y="x2", hue="status", legend=True, kind="kde")
    # sns.displot(data=data_frame_test, x="x1", y="x2", hue="class_name", legend=False, kind="kde")
    sns.pairplot(data=data_frame_test, vars=["x1", "x2", "x3", "x4"], hue="status", kind="kde")
    plt.show()


if __name__ == "__main__":
    main()

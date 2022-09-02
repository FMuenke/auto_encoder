import cv2
import os
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.util import prepare_input

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

    data_frame = {
        "class_name": [],
        "cls_id": [],
        "MAE": [],
        "MSE": []
    }
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

        data = prepare_input(data, input_shape=model.input_shape)
        rec_error_mae = np.mean(np.abs(data - pred))
        rec_error_mse = np.mean(np.sqrt(np.abs(data - pred)))

        data_frame["class_name"].append(cls)
        data_frame["cls_id"].append(class_mapping[cls])
        data_frame["MAE"].append(rec_error_mae)
        data_frame["MSE"].append(rec_error_mse)
    return data_frame, class_mapping


def get_data_sets(ds_path, model_path):
    cfg = Config()
    if os.path.isfile(os.path.join(model_path, "opt.json")):
        cfg.opt = load_dict(os.path.join(model_path, "opt.json"))
    ae = AutoEncoder(model_path, cfg)
    ae.build(False, add_decoder=True)

    data_frame_train, class_mapping = load_data_set(ae, os.path.join(ds_path, "train", "known"))
    save_dict(data_frame_train, os.path.join(model_path, "data_frame_rec_err_train.json"))
    save_dict(class_mapping, os.path.join(model_path, "class_mapping_rec_err_train.json"))

    data_frame_test, class_mapping = load_data_set(ae, os.path.join(ds_path, "test"), class_mapping)
    save_dict(data_frame_test, os.path.join(model_path, "data_frame_rec_err_test.json"))
    save_dict(class_mapping, os.path.join(model_path, "class_mapping_rec_err_test.json"))


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


def main():
    ds_path = "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/TS-DATA-GROUPED"
    model_path = "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/ai_models/AE_128-ENC64-D4-R1"

    # get_data_sets(ds_path, model_path)

    data_frame_train = load_dict(os.path.join(model_path, "data_frame_rec_err_train.json"))
    class_mapping_train = load_dict(os.path.join(model_path, "class_mapping_rec_err_train.json"))

    data_frame_test = load_dict(os.path.join(model_path, "data_frame_rec_err_test.json"))

    unknown_cls_train, unknown_id_train = mark_unknowns(data_frame_train, class_mapping_train)
    unknown_cls, unknown_id = mark_unknowns(data_frame_test, class_mapping_train)
    data_frame_test["status"] = unknown_cls

    auroc = roc_auc_score(unknown_id, np.array(data_frame_test["MSE"]))
    print("[Reconstruction Error] ", 1 - auroc)

    data_frame_test = pd.DataFrame(data_frame_test)
    sns.displot(data=data_frame_test, x="MAE", hue="status", kind="kde")
    plt.show()
    sns.displot(data=data_frame_test, x="MSE", hue="status", kind="kde")
    plt.show()



if __name__ == "__main__":
    main()

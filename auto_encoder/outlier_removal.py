import os
import numpy as np
import pandas as pd

# from umap import UMAP

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.metrics import f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns


def eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test, data_frame_test):
    s = ""
    print("Fitting CLF: {}".format(clf_id))
    clf.fit(x_train, y_train)
    y_cls_only = clf.predict(x_test[y_test != 0, :])
    f1 = f1_score(y_test[y_test != 0], y_cls_only)
    s += "[{} - F1-SCORE] {}\n".format(clf_id, f1)
    proba = clf.predict_proba(x_test)
    max_proba = np.max(proba, axis=1)
    auroc = roc_auc_score(data_frame_test["status_id"], max_proba)
    s += "[{} - AUROC] {}\n".format(clf_id, auroc)
    return s, auroc, f1


def eval_outlier_detector(ood, ood_id, x_train, x_test, data_frame_test):
    s = ""
    print("Fitting OOD: {}".format(ood_id))
    ood.fit(x_train)
    outlier_score = ood.score_samples(x_test)
    auroc = roc_auc_score(data_frame_test["status_id"], outlier_score)
    s += "[{}] {}\n".format(ood_id, auroc)
    return s, auroc


def plot_feature_space(x_train, x_test, data_frame_test, save_path):
    print("[INFO] PCA of feature space:")
    projection = PCA(n_components=4)
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
    plt.savefig(os.path.join(save_path, "pca-dist.png"))
    plt.close()


def eval_outlier_removal(x_train, y_train, x_test, y_test, data_frame_test, save_path):
    s = ""
    s += "[INFO] CLASSIFICATION\n"

    clf_list = [
        [RandomForestClassifier(n_jobs=-1, n_estimators=50), "RANDOM FORREST (50) CLASSIFIER"],
        [RandomForestClassifier(n_jobs=-1), "RANDOM FORREST (100) CLASSIFIER"],
        [RandomForestClassifier(n_jobs=-1, n_estimators=200), "RANDOM FORREST (200) CLASSIFIER"],
        [GaussianProcessClassifier((1.0 * kernels.RBF(1.0)), n_jobs=-1), "GAUSSIAN PROCESS CLASSIFIER"],
        [MLPClassifier(), "MLP (100) CLASSIFIER"],
        [MLPClassifier(hidden_layer_sizes=(256, 128, )), "MLP (256, 128) CLASSIFIER"],
        [KNeighborsClassifier(n_jobs=-1, n_neighbors=5), "KNN (5) CLASSIFIER"],
        [KNeighborsClassifier(n_jobs=-1, n_neighbors=15), "KNN (15) CLASSIFIER"],
    ]
    data = []
    for clf, clf_id in clf_list:
        s_clf, auroc, f1 = eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test, data_frame_test)
        data.append({"name": clf_id, "AUROC": auroc, "F1-Score": f1, "type": "CLS"})
        s += s_clf

    s += "\n[INFO] Fitting outlier removal...\n"

    ood_list = [
        [IsolationForest(n_jobs=-1, contamination=0.01), "ISOLATION FORREST (0.1)"],
        [LocalOutlierFactor(n_jobs=-1, novelty=True, contamination=0.01), "LOCAL OUTLIER FACTOR (0.1)"],
        [EllipticEnvelope(contamination=0.01), "ELLIPTIC ENVELOPE (0.1)"],
        [IsolationForest(n_jobs=-1, contamination=0.25), "ISOLATION FORREST (0.25)"],
        [LocalOutlierFactor(n_jobs=-1, novelty=True, contamination=0.25), "LOCAL OUTLIER FACTOR (0.25)"],
        [EllipticEnvelope(contamination=0.25), "ELLIPTIC ENVELOPE (0.25)"]
    ]

    for ood, ood_id in ood_list:
        s_ood, auroc = eval_outlier_detector(ood, ood_id, x_train, x_test, data_frame_test)
        data.append({"name": ood_id, "AUROC": auroc, "type": "OOD"})
        s += s_ood

    print(s)
    with open(os.path.join(save_path, "outlier-results.txt"), "w") as f:
        f.write(s)

    data_frame = pd.DataFrame(data)
    data_frame.to_csv(os.path.join(save_path, "outlier-results.csv"))

    plot_feature_space(x_train, x_test, data_frame_test, save_path)

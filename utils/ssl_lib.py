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
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


def eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test):
    print("Fitting CLF: {}".format(clf_id))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    return f1, acc


def select_random_subset(x_train, y_train, n_labels):
    n_samples = x_train.shape[0]
    selected = np.random.choice(n_samples, n_labels, replace=False)
    return x_train[selected, :], y_train[selected]


def select_random_subset_by_class(x_train, y_train, n_labels):
    u_classes = np.unique(y_train)
    print("Number of found classes: ", len(u_classes))
    n_labels_per_class = int(n_labels / len(u_classes))
    x_sampled = []
    y_sampled = []
    for u_cls in u_classes:
        x_class = x_train[y_train == u_cls, :]
        y_class = y_train[y_train == u_cls]
        x_class, y_class = select_random_subset(x_class, y_class, n_labels_per_class)
        x_sampled.append(x_class)
        y_sampled.append(y_class)

    x_sampled = np.concatenate(x_sampled, axis=0)
    y_sampled = np.concatenate(y_sampled, axis=0)
    return x_sampled, y_sampled


def cls_test_run(x_train, y_train, x_test, y_test, n_labels):
    clf_list = [
        [RandomForestClassifier(n_jobs=-1, n_estimators=200), "RANDOM FORREST (200) CLASSIFIER"],
        [RandomForestClassifier(n_jobs=-1, n_estimators=2500), "RANDOM FORREST (2500) CLASSIFIER"],
        [MLPClassifier(max_iter=10000), "MLP (100) CLASSIFIER"],
        [MLPClassifier(hidden_layer_sizes=(256, 128,), max_iter=10000), "MLP (256, 128) CLASSIFIER"],
        [LogisticRegression(n_jobs=-1), "LR"]
    ]

    data_frame = []
    x_train, y_train = select_random_subset_by_class(x_train, y_train, n_labels)
    for clf, clf_id in clf_list:
        f1, acc = eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test)
        data_frame.append({
            "clf": clf_id,
            "F1-Score": f1,
            "Accuracy": acc,
            "n_labels": n_labels,
        })
    data_frame = pd.DataFrame(data_frame)
    return data_frame


def eval_semi_supervised_classification(x_train, y_train, x_test, y_test, save_path):
    data_frame = []
    for n_labels in [500, 2500, 10000]:
        data_frame_p = cls_test_run(x_train, y_train, x_test, y_test, n_labels)
        data_frame.append(data_frame_p)
    data_frame = pd.concat(data_frame, ignore_index=True)
    sns.lineplot(data=data_frame, x="n_labels", y="Accuracy", hue="clf")
    plt.savefig(os.path.join(save_path, "clf_accuracy.png"))
    plt.close()

    sns.lineplot(data=data_frame, x="n_labels", y="F1-Score", hue="clf")
    plt.savefig(os.path.join(save_path, "clf_accuracy.png"))
    plt.close()

    data_frame.to_csv(os.path.join(save_path, "classifier_results.csv"))

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from utils.mlp_classifier import TfMlp


def eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test):
    print("Fitting CLF: {}".format(clf_id))
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)

    labels_list = [i for i in range(y_pred.shape[1])]
    f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average="weighted")
    acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
    acc_5 = top_k_accuracy_score(y_test, y_pred, k=5, labels=labels_list)
    return f1, acc, acc_5


def select_random_subset(x_train, y_train, n_labels):
    n_samples = x_train.shape[0]
    selected = np.random.choice(n_samples, n_labels, replace=False)
    return x_train[selected, :], y_train[selected]


def select_random_subset_by_class(x_train, y_train, n_labels):
    n_samples = x_train.shape[0]
    if n_labels >= n_samples:
        print("No sampling!")
        return x_train, y_train
    u_classes = np.unique(y_train)
    print("Number of found classes: {}. Selecting: {}".format(len(u_classes), n_labels))
    n_labels_per_class = int(n_labels / len(u_classes))
    x_sampled = []
    y_sampled = []
    for u_cls in u_classes:
        x_class = x_train[y_train == u_cls, :]
        y_class = y_train[y_train == u_cls]
        if y_class.shape[0] > n_labels_per_class:
            x_class, y_class = select_random_subset(x_class, y_class, n_labels_per_class)
        else:
            print("[WARNING] Not enough labels ({}) in class to sample {} instances".format(
                y_class.shape[0], n_labels_per_class))
        x_sampled.append(x_class)
        y_sampled.append(y_class)

    x_sampled = np.concatenate(x_sampled, axis=0)
    y_sampled = np.concatenate(y_sampled, axis=0)
    return x_sampled, y_sampled


def cls_test_run(x_train, y_train, x_test, y_test, n_labels, run_id):
    clf_list = [
        [LogisticRegression(max_iter=10000, n_jobs=-1), "LR"],
        [TfMlp(x_train.shape[1], 100, [512, 256], dropout_rate=0.75), "TF-MLP (512, 256) drp=0.75"],
        # [TfMlp(x_train.shape[1], 100, [1024], dropout_rate=0.75), "TF-MLP (1024) drp=0.75"],
        # [neighbors.NearestCentroid(), "NC"],
        [neighbors.KNeighborsClassifier(), "KNN"]
    ]

    data_frame = []
    x_train, y_train = select_random_subset_by_class(x_train, y_train, n_labels)
    for clf, clf_id in clf_list:
        f1, acc, acc_5 = eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test)
        data_frame.append({
            "clf": clf_id,
            "F1-Score": f1,
            "Accuracy": acc,
            "Top 5 Accuracy": acc_5,
            "n_labels": n_labels,
            "run": run_id,
        })
    data_frame = pd.DataFrame(data_frame)
    return data_frame


def eval_semi_supervised_classification(x_train, y_train, x_test, y_test, save_path, direct_features):
    data_frame = []

    u_classes = np.unique(y_train)
    n_classes = len(u_classes)

    for n_labels in [n_classes*4, n_classes*10, n_classes*20, n_classes*40, n_classes*100]:
        for i in range(1):
            data_frame_p = cls_test_run(x_train, y_train, x_test, y_test, n_labels, i)
            data_frame.append(data_frame_p)
    data_frame = pd.concat(data_frame, ignore_index=True)

    if direct_features:
        ds_ident = "_DIRECT"
    else:
        ds_ident = ""

    sns.lineplot(data=data_frame, x="n_labels", y="Accuracy", hue="clf")
    plt.savefig(os.path.join(save_path, "clf_accuracy{}.png".format(ds_ident)))
    plt.close()

    sns.lineplot(data=data_frame, x="n_labels", y="F1-Score", hue="clf")
    plt.savefig(os.path.join(save_path, "clf_f1_score{}.png".format(ds_ident)))
    plt.close()

    data_frame.to_csv(os.path.join(save_path, "classifier_results{}.csv".format(ds_ident)))
    print(data_frame)

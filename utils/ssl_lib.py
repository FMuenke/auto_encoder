import os
import numpy as np
import pandas as pd


import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


class TfMlp:
    def __init__(self, input_dim, n_classes, layer_dims, train=True, dropout_rate=0.25):
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = 10e-5

        x_in = keras.layers.Input(shape=input_dim)

        x = self.add_mlp(x_in, layer_dims)
        x = keras.layers.Dense(n_classes, kernel_regularizer="l2")(x)
        y = keras.layers.Softmax()(x)

        self.model = keras.models.Model(inputs=x_in, outputs=y)
        if train:
            self.model.compile(
                optimizer=tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.learning_rate*0.1),
                loss="categorical_crossentropy",
                metrics=[
                    "accuracy",
                    keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy')
                ]
            )

    def add_mlp(self, x, hidden_units):
        for units in hidden_units:
            x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)
        return x

    def fit(self, x, y):
        x_trn, x_val, y_trn, y_val = train_test_split(x, y, stratify=y, test_size=0.25)
        self.model.fit(
            x_trn,
            tf.one_hot(y_trn, self.n_classes),
            batch_size=200,
            epochs=10000,
            validation_data=(x_val, tf.one_hot(y_val, self.n_classes)),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=1000, restore_best_weights=True)
            ],
            verbose=0,
            # use_multiprocessing=False
        )

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = np.array(y_pred)
        return np.argmax(y_pred, axis=1)


def get_logistic_regression_optimizer(n_labels):
    if n_labels <= 400:
        n_splits = 2
    else:
        n_splits = 3
    clf = LogisticRegressionCV(n_jobs=-1, max_iter=100000, cv=n_splits)
    return clf


def eval_classifier(clf, clf_id, x_train, y_train, x_test, y_test):
    print("Fitting CLF: {}".format(clf_id))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    # acc_5 = top_k_accuracy_score(y_test, y_pred, k=5)
    return f1, acc


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
        x_class, y_class = select_random_subset(x_class, y_class, n_labels_per_class)
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
        [neighbors.NearestCentroid(), "NC"],
        [neighbors.KNeighborsClassifier(), "KNN"]
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
            "run": run_id,
        })
    data_frame = pd.DataFrame(data_frame)
    return data_frame


def eval_semi_supervised_classification(x_train, y_train, x_test, y_test, save_path, direct_features):
    data_frame = []

    for n_labels in [65*4, 65*8, 65*20, 65*40]:
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

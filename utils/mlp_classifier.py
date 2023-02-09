import os.path

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import numpy as np

from sklearn.model_selection import train_test_split


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
            callbacks=[keras.callbacks.EarlyStopping(patience=1000, restore_best_weights=True)],
            verbose=0,
            # use_multiprocessing=False
        )

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = np.array(y_pred)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, x):
        y_pred = self.model.predict(x)
        return np.array(y_pred)

    def save(self, path_to_store):
        if os.path.isdir(path_to_store):
            path_to_store = os.path.join(path_to_store, "mlp_weights.hdf5")
        self.model.save(path_to_store)

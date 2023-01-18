import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy

import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger


from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.essentials import add_classification_head
from auto_encoder.data_generator import HybridDataGenerator

from auto_encoder.util import check_n_make_dir
from auto_encoder.util import prepare_input

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class HybridImageClassifier(AutoEncoder):
    def __init__(self, model_folder, cfg, class_mapping):
        super(HybridImageClassifier, self).__init__(model_folder, cfg)
        self.metric_to_track = "val_loss"
        self.class_mapping = class_mapping

    def inference(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.array(res[0])[0, :]
        res = np.argmax(res)
        return res

    def encode(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.array(res)
        return res

    def build(self, compile_model=True, add_decoder=True):
        x_input, bottleneck, output = self.get_backbone()

        classification = add_classification_head(
            bottleneck, n_classes=len(self.class_mapping), hidden_units=[512, 256], dropout_rate=0.75)

        self.model = Model(inputs=x_input, outputs=[classification, output])
        self.load()
        if compile_model:
            self.model.compile(
                loss={
                    "conv_transpose_5": "mean_squared_error",
                    "clf_final": CategoricalCrossentropy(from_logits=True)
                },
                metrics={
                    "clf_final": keras.metrics.CategoricalAccuracy(name="accuracy")
                },
                optimizer=self.optimizer)

    def fit(self, tag_set_train, tag_set_test, augmentations):
        print("[INFO] Training with {} / Testing with {}".format(len(tag_set_train), len(tag_set_test)))

        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        check_n_make_dir(self.model_folder)

        self.batch_size = np.min([len(tag_set_train), len(tag_set_test), self.batch_size])

        training_generator = HybridDataGenerator(
            tag_set_train,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            augmentations=augmentations,
            class_mapping=self.class_mapping
        )
        validation_generator = HybridDataGenerator(
            tag_set_test,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            class_mapping=self.class_mapping,
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-final.hdf5"),
            monitor=self.metric_to_track,
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True
        )

        patience = 32
        reduce_lr = ReduceLROnPlateau(monitor=self.metric_to_track, verbose=1, patience=int(patience*0.5))
        early_stop = EarlyStopping(monitor=self.metric_to_track, patience=patience, verbose=1)
        csv_logger = CSVLogger(filename=os.path.join(self.model_folder, "logs.csv"))

        callback_list = [checkpoint, reduce_lr, early_stop, csv_logger]

        print("[INFO] Training started. Results: {}".format(self.model_folder))
        history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            verbose=1,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

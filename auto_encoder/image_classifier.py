import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers

import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger


from auto_encoder.backbone.essentials import add_classification_head
from auto_encoder.backbone.encoder import get_encoder
from auto_encoder.data_generator import ClassificationDataGenerator

from auto_encoder.util import check_n_make_dir
from auto_encoder.util import prepare_input

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class ImageClassifier:
    def __init__(self, model_folder, cfg, class_mapping):
        self.metric_to_track = "val_loss"
        self.model_folder = model_folder
        self.class_mapping = class_mapping

        self.init_learning_rate = cfg.opt["init_learning_rate"]
        self.input_shape = cfg.opt["input_shape"]
        self.backbone = cfg.opt["backbone"]
        self.embedding_size = cfg.opt["embedding_size"]
        self.depth = cfg.opt["depth"]
        self.resolution = cfg.opt["resolution"]
        self.embedding_type = cfg.opt["embedding_type"]
        self.embedding_activation = cfg.opt["embedding_activation"]
        self.drop_rate = cfg.opt["drop_rate"]
        self.dropout_structure = cfg.opt["dropout_structure"]
        self.freeze = cfg.opt["freeze"]
        self.scale = cfg.opt["scale"]

        self.model = None

        optimizer = None
        if "optimizer" in cfg.opt:
            if "adam" == cfg.opt["optimizer"]:
                optimizer = optimizers.Adam(learning_rate=cfg.opt["init_learning_rate"])
            if "ada_delta" == cfg.opt["optimizer"]:
                optimizer = optimizers.Adadelta()
        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=cfg.opt["init_learning_rate"])
        self.optimizer = optimizer

        self.batch_size = 1
        if "batch_size" in cfg.opt:
            self.batch_size = cfg.opt["batch_size"]
        self.epochs = 10000

    def inference(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.argmax(np.array(res))
        conf = np.max(res)
        return res, conf

    def get_backbone(self):
        x_in, output = get_encoder(
            backbone=self.backbone,
            input_shape=self.input_shape,
            embedding_size=self.embedding_size,
            embedding_type=self.embedding_type,
            embedding_activation=self.embedding_activation,
            depth=self.depth,
            scale=self.scale,
            resolution=self.resolution,
            drop_rate=self.drop_rate,
            dropout_structure=self.dropout_structure,
        )
        output = add_classification_head(
            output,
            n_classes=len(self.class_mapping),
            hidden_units=[],
            dropout_rate=0.0
        )
        return x_in, output

    def build(self, compile_model=True):
        x_in, output = self.get_backbone()

        self.model = keras.Model(x_in, output)
        if self.freeze:
            for layer in self.model.layers:
                if str(layer.name).startswith("clf_"):
                    continue
                layer.trainable = False

        self.load()
        if compile_model:
            self.model.compile(
                optimizer=self.optimizer,
                loss=[
                    keras.losses.CategoricalCrossentropy(from_logits=True),
                ],
                metrics=[
                    keras.metrics.CategoricalAccuracy(name="accuracy"),
                    keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )

    def load(self):
        model_path = None
        if os.path.isfile(os.path.join(self.model_folder, "weights-final.hdf5")):
            model_path = os.path.join(self.model_folder, "weights-final.hdf5")
        elif os.path.isdir(self.model_folder):
            pot_models = sorted(os.listdir(self.model_folder))
            for model in pot_models:
                if model.lower().endswith((".hdf5", ".h5")):
                    model_path = os.path.join(self.model_folder, model)
        if model_path is not None:
            print("[INFO] Model-Weights are loaded from: {}".format(model_path))
            self.model.load_weights(model_path, by_name=True)
        else:
            print("[INFO] No Weights were found")

    def fit(self, tag_set_train, tag_set_test, augmentations):
        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        check_n_make_dir(self.model_folder)

        print("[INFO] Training with {} / Testing with {}".format(len(tag_set_train), len(tag_set_test)))

        self.batch_size = np.min([len(tag_set_train), len(tag_set_test), self.batch_size])

        training_generator = ClassificationDataGenerator(
            tag_set_train,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            augmentations=augmentations,
            class_mapping=self.class_mapping,
        )
        validation_generator = ClassificationDataGenerator(
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

        print("[INFO] Training started.")
        history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            verbose=1,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

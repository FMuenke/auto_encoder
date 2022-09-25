import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger


from auto_encoder.backbone import Backbone
from auto_encoder.data_generator import DataGenerator

from auto_encoder.util import check_n_make_dir


try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     physical_devices[0],
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
    # )

except Exception as e:
    print(e)
    print("ATTENTION: GPU IS NOT USED....")


from auto_encoder.util import prepare_input


class AutoEncoder:
    def __init__(self, model_folder, cfg):
        self.metric_to_track = "val_loss"
        self.model_folder = model_folder

        self.init_learning_rate = cfg.opt["init_learning_rate"]
        self.input_shape = cfg.opt["input_shape"]
        self.backbone = cfg.opt["backbone"]
        self.embedding_size = cfg.opt["embedding_size"]
        self.depth = cfg.opt["depth"]
        self.resolution = cfg.opt["resolution"]
        if "loss" in cfg.opt:
            self.loss_type = cfg.opt["loss"]

        self.model = None

        optimizer = None
        if "optimizer" in cfg.opt:
            if "adam" == cfg.opt["optimizer"]:
                optimizer = optimizers.Adam(lr=cfg.opt["init_learning_rate"])
            if "ada_delta" == cfg.opt["optimizer"]:
                optimizer = optimizers.Adadelta()
        if optimizer is None:
            optimizer = optimizers.Adam(lr=cfg.opt["init_learning_rate"])
        self.optimizer = optimizer

        if "batch_size" in cfg.opt:
            self.batch_size = cfg.opt["batch_size"]
        else:
            self.batch_size = 1
        self.epochs = 10000

    def inference(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.array(res)
        return res

    def build(self, compile_model=True, add_decoder=True):
        backbone = Backbone(
            self.backbone,
            embedding_size=self.embedding_size,
            depth=self.depth,
            resolution=self.resolution,
            loss_type=self.loss_type
        )
        x_input, y = backbone.build(self.input_shape, add_decoder=add_decoder)

        self.model = Model(inputs=x_input, outputs=y)
        # print(self.model.summary())

        self.load()
        if compile_model:
            self.model.compile(loss=backbone.loss(), optimizer=self.optimizer, metrics=backbone.metric())

    def load(self):
        model_path = None
        if os.path.isdir(self.model_folder):
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

        self.batch_size = np.min([len(tag_set_train), len(tag_set_test), self.batch_size])

        training_generator = DataGenerator(
            tag_set_train,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            augmentations=augmentations,
        )
        validation_generator = DataGenerator(
            tag_set_test,
            image_size=self.input_shape,
            batch_size=self.batch_size,
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-final.hdf5"),
            monitor=self.metric_to_track,
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        patience = 64
        reduce_lr = ReduceLROnPlateau(monitor=self.metric_to_track, verbose=1, patience=int(patience*0.5))
        # reduce_lr = CosineDecayRestarts(initial_learning_rate=self.init_learning_rate, first_decay_steps=1000)
        early_stop = EarlyStopping(monitor=self.metric_to_track, patience=patience, verbose=1)
        csv_logger = CSVLogger(filename=os.path.join(self.model_folder, "logs.csv"))

        callback_list = [checkpoint, reduce_lr, early_stop, csv_logger]

        history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            verbose=1,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)




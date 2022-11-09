import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger


from auto_encoder.residual import residual_auto_encoder, asymmetric_residual_auto_encoder
from auto_encoder.fully_connected import fully_connected_auto_encoder
from auto_encoder.asymmetric_auto_encoder import asymmetric_auto_encoder
from auto_encoder.data_generator import DataGenerator

from auto_encoder.util import check_n_make_dir
from auto_encoder.util import prepare_input

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# try:
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     physical_devices[0],
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
    # )

# except Exception as e:
    # print(e)
    # print("ATTENTION: GPU IS NOT USED....")


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

        if "embedding_type" not in cfg.opt:
            cfg.opt["embedding_type"] = "flatten"
        if "embedding_activation" not in cfg.opt:
            cfg.opt["embedding_activation"] = "linear"
        if "drop_rate" not in cfg.opt:
            cfg.opt["drop_rate"] = 0.0
        self.embedding_type = cfg.opt["embedding_type"]
        self.embedding_activation = cfg.opt["embedding_activation"]
        self.drop_rate = cfg.opt["drop_rate"]

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

    def get_backbone(self, add_decoder):
        if self.backbone in ["resnet", "residual"]:
            x_input, bottleneck, output = residual_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                embedding_activation=self.embedding_activation,
                depth=self.depth,
                resolution=self.resolution,
                drop_rate=self.drop_rate,
            )
        elif self.backbone in ["asym-residual", "asymmetric-residual"]:
            x_input, bottleneck, output = asymmetric_residual_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                embedding_activation=self.embedding_activation,
                depth=self.depth,
                resolution=self.resolution,
                drop_rate=self.drop_rate,
            )
        elif self.backbone in ["fully_connected", "fc"]:
            x_input, bottleneck, output = fully_connected_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_activation=self.embedding_activation,
                drop_rate=self.drop_rate
            )
        elif self.backbone in ["asymetric"]:
            x_input, bottleneck, output = asymmetric_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                embedding_activation=self.embedding_activation,
                drop_rate=self.drop_rate,
            )
        else:
            raise ValueError("{} Backbone was not recognised".format(self.backbone))

        if add_decoder:
            return x_input, output
        else:
            return x_input, bottleneck

    def build(self, compile_model=True, add_decoder=True):
        x_input, y = self.get_backbone(add_decoder)

        self.model = Model(inputs=x_input, outputs=y)
        self.load()
        if compile_model:
            self.model.compile(loss="mean_squared_error", optimizer=self.optimizer, metrics=["mse"])

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
            augmentations=augmentations,
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-final.hdf5"),
            monitor=self.metric_to_track,
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True
        )

        patience = 128
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
            verbose=0,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)




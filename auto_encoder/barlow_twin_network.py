import os
import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.residual import make_residual_simple_siamese
from auto_encoder.barlow_twin_network_engine import BarlowTwin, BarlowLoss
from auto_encoder.simple_siamese_data_generator import DataGenerator

from auto_encoder.util import check_n_make_dir, prepare_input


class BarlowTwinNetwork(AutoEncoder):
    def __init__(self, model_folder, cfg):
        super(BarlowTwinNetwork, self).__init__(model_folder, cfg)
        self.metric_to_track = "loss"

    def get_backbone(self):
        if self.backbone in ["resnet", "residual"]:
            encoder, decoder = make_residual_simple_siamese(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                embedding_activation=self.embedding_activation,
                depth=self.depth,
                scale=self.scale,
                resolution=self.resolution,
                noise=self.embedding_noise,
                drop_rate=self.drop_rate,
                dropout_structure=self.dropout_structure
            )
        else:
            raise ValueError("{} Backbone was not recognised".format(self.backbone))

        return BarlowTwin(encoder)

    def encode(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.array(res)
        return res

    def build(self, compile_model=True, add_decoder=True):
        self.model = self.get_backbone()
        self.model(np.zeros((1, self.input_shape[0], self.input_shape[1], 3)))

        self.load()
        if compile_model:
            self.model.compile(optimizer=self.optimizer, loss=BarlowLoss(self.batch_size))

    def fit(self, tag_set_train, tag_set_test, augmentations):
        print("[INFO] Training with {} / Testing with {}".format(len(tag_set_train), len(tag_set_test)))
        print("[INFO] Combining to {}".format(len(tag_set_train + tag_set_test)))

        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        check_n_make_dir(self.model_folder)

        self.batch_size = np.min([len(tag_set_train), len(tag_set_test), self.batch_size])

        training_generator = DataGenerator(
            tag_set_train + tag_set_test,
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

        patience = 5
        reduce_lr = ReduceLROnPlateau(monitor=self.metric_to_track, verbose=1, patience=int(patience*0.5))
        # reduce_lr = CosineDecayRestarts(initial_learning_rate=self.init_learning_rate, first_decay_steps=1000)
        early_stop = EarlyStopping(monitor=self.metric_to_track, patience=patience, verbose=1)
        csv_logger = CSVLogger(filename=os.path.join(self.model_folder, "logs.csv"))

        callback_list = [checkpoint, reduce_lr, early_stop, csv_logger]

        print("[INFO] Training started. Results: {}".format(self.model_folder))
        history = self.model.fit(
            x=training_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            verbose=1,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)



from tensorflow import keras
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model

from auto_encoder.auto_encoder import AutoEncoder


class ClusteringAutoEncoder(AutoEncoder):
    def __init__(self, model_folder, opt):
        super(ClusteringAutoEncoder, self).__init__(model_folder, opt)
        self.n_clusters = opt["n_clusters"]
        self.metric_to_track = "val_loss"

    def build(self, compile_model=True, add_decoder=True):
        x_input, bottleneck, output = self.get_backbone()
        if add_decoder:
            cluster = keras.layers.Dense(self.n_clusters, activation="softmax")(bottleneck)
            y = [output, cluster]
        else:
            y = bottleneck

        self.model = Model(inputs=x_input, outputs=y)
        self.load()
        if compile_model:
            self.model.compile(loss=["mean_squared_error", ], optimizer=self.optimizer, metrics=["mse"])
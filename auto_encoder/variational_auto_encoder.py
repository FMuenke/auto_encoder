import os
import numpy as np
import matplotlib.pyplot as plt

from auto_encoder.auto_encoder import AutoEncoder
from auto_encoder.residual import residual_variational_auto_encoder
from auto_encoder.fully_connected import variational_fully_connected_auto_encoder
from auto_encoder.linear import linear_variational_auto_encoder
from auto_encoder.variational_auto_encoder_engine import VariationalAutoEncoderEngine

from auto_encoder.util import prepare_input


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, model_folder, cfg):
        super(VariationalAutoEncoder, self).__init__(model_folder, cfg)

        self.metric_to_track = "val_loss"

    def get_backbone(self):
        if self.backbone in ["resnet", "residual"]:
            encoder, decoder = residual_variational_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                depth=self.depth,
                scale=self.scale,
                resolution=self.resolution,
            )
        elif self.backbone in ["linear", "lin"]:
            encoder, decoder = linear_variational_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                depth=self.depth,
                resolution=self.resolution,
            )
        elif self.backbone in ["fully_connected", "fc"]:
            encoder, decoder = variational_fully_connected_auto_encoder(
                input_shape=self.input_shape,
                embedding_size=self.embedding_size,
            )
        else:
            raise ValueError("{} Backbone was not recognised".format(self.backbone))

        return VariationalAutoEncoderEngine(encoder, decoder)

    def build(self, compile_model=True, add_decoder=True):
        self.model = self.get_backbone()
        self.model(np.zeros((1, self.input_shape[0], self.input_shape[1], 3)))
        self.load()
        if compile_model:
            self.model.compile(optimizer=self.optimizer)

    def inference(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.predict_on_batch(data)
        res = np.array(res)
        return res

    def encode(self, data):
        data = prepare_input(data, self.input_shape)
        data = np.expand_dims(data, axis=0)
        res = self.model.encoder.predict_on_batch(data)
        z_mean = np.array(res[0])
        # z_log_var = np.array(res[1])
        # res = np.concatenate([z_mean, z_log_var], axis=1)
        return z_mean

    def plot_latent_space(self, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = self.input_shape[1]
        scale = 1.0
        n = 30
        figure = np.zeros((self.input_shape[1] * n, self.input_shape[1] * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space

        grid_x = np.linspace(0, self.embedding_size, n)
        grid_y = np.linspace(0, self.embedding_size, n)

        print("[INFO] Predicting Latent-Space...")
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array(np.zeros((1, self.embedding_size)))
                xi = int(xi)
                yi = int(yi)
                if yi < xi:
                    z_sample[0, yi:xi] = 1
                else:
                    z_sample[0, xi:yi] = -1
                x_decoded = self.model.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(self.input_shape[0], self.input_shape[1], self.input_shape[2])
                digit = np.mean(digit, axis=2)
                figure[
                    i * digit_size: (i + 1) * self.input_shape[0],
                    j * digit_size: (j + 1) * self.input_shape[1],
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig(os.path.join(self.model_folder, "latent_space.png"))
        plt.close()

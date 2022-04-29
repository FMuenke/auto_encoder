from tensorflow.keras import applications
import tensorflow as tf
from tensorflow.keras import layers


class Backbone:
    def __init__(self, backbone_type, embedding_size, loss_type="mse", weights="imagenet"):
        self.backbone_type = backbone_type
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        self.weights = weights

    def loss(self):
        if self.loss_type in ["mean_squared_error", "mse"]:
            return "mean_squared_error"

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def metric(self):
        return ["mse"]

    def build_encoder(self, input_shape):
        if self.backbone_type in ["keras-resnet50", "keras-resnet"]:
            base_model = applications.resnet50.ResNet50(
                weights=self.weights,
                include_top=False,
                input_shape=(input_shape[0], input_shape[1], input_shape[2]),
            )
            return base_model.input, base_model.output

        if self.backbone_type in ["basic"]:
            input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
            # Block 1
            x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', name='conv_1')(input_layer)
            x = layers.BatchNormalization(name='bn_1')(x)
            x = layers.LeakyReLU(name='lrelu_1')(x)
            # Block 2
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_2')(x)
            x = layers.BatchNormalization(name='bn_2')(x)
            x = layers.LeakyReLU(name='lrelu_2')(x)
            # Block 3
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_3')(x)
            x = layers.BatchNormalization(name='bn_3')(x)
            x = layers.LeakyReLU(name='lrelu_3')(x)
            # Block 4
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_4')(x)
            x = layers.BatchNormalization(name='bn_4')(x)
            x = layers.LeakyReLU(name='lrelu_4')(x)
            # Block 5
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_5')(x)
            x = layers.BatchNormalization(name='bn_5')(x)
            x = layers.LeakyReLU(name='lrelu_5')(x)
            return input_layer, x

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

    def build_decoder(self, inputs):
        x = layers.Dense(8*8*64, name='dense_1')(inputs)
        x = tf.reshape(x, [-1, 8, 8, 64], name='Reshape_Layer')

        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_1')(x)
        x = layers.BatchNormalization(name='bn_d1')(x)
        x = layers.LeakyReLU(name='lrelu_d1')(x)

        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='conv_transpose_2')(x)
        x = layers.BatchNormalization(name='bn_d2')(x)
        x = layers.LeakyReLU(name='lrelu_d2')(x)

        x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
        x = layers.BatchNormalization(name='bn_d3')(x)
        x = layers.LeakyReLU(name='lrelu_d3')(x)

        x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4')(x)
        x = layers.BatchNormalization(name='bn_d4')(x)
        x = layers.LeakyReLU(name='lrelu_d4')(x)
        outputs = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='sigmoid', name='conv_transpose_5')(x)
        return outputs

    def build(self, input_shape, add_decoder=True):
        x_input, latent = self.build_encoder(input_shape)
        latent_flat = layers.Flatten()(latent)
        bottleneck = layers.Dense(self.embedding_size)(latent_flat)
        if add_decoder:
            output = self.build_decoder(bottleneck)
            return x_input, output
        else:
            return x_input, bottleneck


import tensorflow as tf
from tensorflow.keras import layers


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def create_dense_encoder(x, embedding_size):
    bottleneck = layers.Dense(embedding_size)(x)
    return bottleneck


class Embedding:
    def __init__(self,
                 embedding_size,
                 embedding_type,
                 activation,
                 drop_rate,
                 noise,
                 skip,
                 mode="2d"
                 ):
        self.embedding_size = embedding_size
        self.embedding_type = embedding_type
        self.activation = activation

        self.drop_rate = drop_rate
        self.noise = noise

        self.skip = skip

        self.mode = mode

    def create_bayesian_encoder(self, x, n_inferences=16):
        if self.mode == "2d":
            batch_size, h, w, n_features = x.shape
            n_samples = h * w
        elif self.mode == "1d":
            batch_size, n_samples, n_features = x.shape
        else:
            raise Exception("Unknown Mode - {} -".format(self.mode))
        x = tf.reshape(x, (-1, n_samples, n_features))
        x = tf.repeat(x, n_inferences, axis=1)
        x = layers.TimeDistributed(layers.Dropout(self.drop_rate))(x, training=True)
        x = layers.TimeDistributed(layers.Dense(self.embedding_size))(x)
        x = layers.GlobalAveragePooling1D()(x)
        return x

    def add_conv_layer(self, x, n_filters, kernel_size):
        if self.mode == "2d":
            x = layers.Conv2D(n_filters, (kernel_size, kernel_size))(x)
        elif self.mode == "1d":
            x = layers.Conv1D(n_filters, kernel_size)(x)
        else:
            raise Exception("Unknown Mode - {} -".format(self.mode))
        return x

    def add_pooling_op(self, x, pooling):
        if pooling == "max" and self.mode == "2d":
            return layers.GlobalMaxPool2D()(x)
        elif pooling == "max" and self.mode == "1d":
            return layers.GlobalMaxPool1D()(x)
        elif pooling == "avg" and self.mode == "2d":
            return layers.GlobalAveragePooling2D()(x)
        elif pooling == "avg" and self.mode == "1d":
            return layers.GlobalAveragePooling1D()(x)
        else:
            raise Exception("Unknown Pooling - {}/{} -".format(self.mode, pooling))

    def build(self, x):

        x_pass = x

        if self.embedding_type == "flatten":
            latent = layers.Flatten()(x)
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "glob_avg":
            latent = self.add_pooling_op(x, "avg")
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "glob_max":
            latent = self.add_pooling_op(x, "max")
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "conv":
            bottleneck = self.add_conv_layer(x, self.embedding_size, kernel_size=1)
        elif self.embedding_type == "bayesian":
            bottleneck = self.create_bayesian_encoder(x)
        else:
            raise Exception("Unknown Embedding Type - {} -".format(self.embedding_type))

        if self.activation == "leaky_relu":
            bottleneck = layers.LeakyReLU()(bottleneck)
            bottleneck = layers.BatchNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "relu":
            bottleneck = layers.ReLU()(bottleneck)
            bottleneck = layers.BatchNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "linear":
            bottleneck = layers.BatchNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "softmax":
            bottleneck = layers.Softmax()(bottleneck)
        elif self.activation == "sigmoid":
            bottleneck = layers.Activation("sigmoid")(bottleneck)

        if self.noise > 0:
            print("[INFO] Gaussian Noise Added.")
            bottleneck = layers.GaussianNoise(1.0 * self.noise)(bottleneck)

        if self.skip:
            print("[INFO] SKIPP CONNECTION ACTIVE")
            x_pass = self.add_conv_layer(x_pass, self.embedding_size, kernel_size=1)
            x_pass = relu_bn(x_pass)
            x_pass = layers.Flatten()(x_pass)
            x_pass = layers.Dropout(self.drop_rate)(x_pass)
            x_pass = layers.Concatenate()([bottleneck, x_pass])
        else:
            if self.drop_rate > 0:
                bottleneck = layers.Dropout(self.drop_rate)(bottleneck)
            x_pass = bottleneck

        if self.embedding_type == "conv":
            bottleneck = self.add_pooling_op(x_pass, "avg")

        return bottleneck, x_pass

    def transform_to_feature_maps(self, x, f_map_height, f_map_width, n_features):
        if self.embedding_type == "conv":
            x = layers.Conv2D(n_features, (1, 1))(x)
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Dense(int(f_map_height * f_map_width * n_features), name='dense_1')(x)
            x = layers.LeakyReLU()(x)
            x = tf.reshape(x, [-1, int(f_map_height), int(f_map_width), n_features], name='Reshape_Layer')
        return x



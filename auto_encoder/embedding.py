import tensorflow as tf
from tensorflow.keras import layers, regularizers

from auto_encoder.essentials import relu_bn, mlp


def create_dense_encoder(x, embedding_size):
    bottleneck = layers.Dense(embedding_size)(x)
    return bottleneck


class Embedding:
    def __init__(self,
                 embedding_size,
                 embedding_type,
                 activation,
                 drop_rate,
                 dropout_structure,
                 mode="2d"
                 ):
        self.embedding_size = embedding_size
        self.embedding_type = embedding_type
        self.activation = activation

        self.drop_rate = drop_rate
        self.dropout_structure = dropout_structure

        self.mode = mode

    def add_conv_layer(self, x, n_filters, kernel_size):
        if self.mode == "2d":
            x = layers.Conv2D(n_filters, (kernel_size, kernel_size))(x)
        elif self.mode == "1d":
            x = layers.Conv1D(n_filters, kernel_size)(x)
        else:
            raise Exception("Unknown Mode - {} -".format(self.mode))
        return x

    def add_local_dropout(self, x, drop_rate):
        if self.mode == "2d":
            batch_size, fmh, fmw, n_f = x.shape
            x = layers.Dropout(drop_rate, noise_shape=[batch_size, fmh, fmw, 1])(x)
        elif self.mode == "1d":
            batch_size, fm, n_f = x.shape
            x = layers.Dropout(drop_rate, noise_shape=[batch_size, fm, 1])(x)
        else:
            raise Exception("Unknown Mode - {} -".format(self.mode))
        return x

    def add_dropout(self, x, drop_rate):
        if self.dropout_structure == "general":
            return layers.Dropout(drop_rate)(x)
        elif self.dropout_structure == "local":
            return self.add_local_dropout(x, drop_rate)
        elif self.dropout_structure == "spatial":
            if self.mode == "1d":
                return layers.SpatialDropout1D(drop_rate)(x)
            elif self.mode == "2d":
                return layers.SpatialDropout2D(drop_rate)(x)
            else:
                raise Exception("Unknown Mode - {} -".format(self.mode))
        else:
            raise Exception("Unknown Dropout Structure - {} -".format(self.dropout_structure))

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

        if self.drop_rate > 0.0:
            x = self.add_dropout(x, self.drop_rate)

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
            latent = self.add_conv_layer(x, self.embedding_size, kernel_size=1)
            bottleneck = self.add_pooling_op(latent, "avg")
        elif self.embedding_type == "glob_avg_mlp":
            latent = self.add_pooling_op(x, "avg")
            latent = mlp(latent, [2 * self.embedding_size], dropout_rate=0.25)
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "direct_avg":
            self.activation = "none"
            bottleneck = self.add_pooling_op(x, "avg")
        elif self.embedding_type == "projection_avg":
            latent = self.add_pooling_op(x, "avg")
            w_decay = 0.0005
            latent = layers.Dense(
                self.embedding_size,
                use_bias=False,
                kernel_regularizer=regularizers.l2(w_decay)
            )(latent)
            latent = relu_bn(latent)
            bottleneck = layers.Dense(
                self.embedding_size,
                use_bias=False,
                kernel_regularizer=regularizers.l2(w_decay)
            )(latent)
        else:
            raise Exception("Unknown Embedding Type - {} -".format(self.embedding_type))

        if self.activation == "leaky_relu":
            bottleneck = layers.LeakyReLU(name="bottleneck_leaky_relu")(bottleneck)
            bottleneck = layers.LayerNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "relu":
            bottleneck = layers.ReLU(name="bottleneck_relu")(bottleneck)
            bottleneck = layers.LayerNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "linear":
            bottleneck = layers.LayerNormalization(name="bottleneck_bn")(bottleneck)
        elif self.activation == "softmax":
            bottleneck = layers.Softmax(name="bottleneck_softmax")(bottleneck)
        elif self.activation == "sigmoid":
            bottleneck = layers.Activation("sigmoid", name="bottleneck_sigmoid")(bottleneck)
        elif self.activation == "none":
            pass

        return bottleneck


def transform_to_feature_maps(x, f_map_height, f_map_width, n_features):
    x = layers.Dense(int(f_map_height * f_map_width * n_features), name='dense_transform')(x)
    x = relu_bn(x)
    x = tf.reshape(x, [-1, int(f_map_height), int(f_map_width), n_features], name='reshape_to_feature_map')
    return x


def transform_to_features(x, f_count, n_features):
    x = layers.Dense(int(f_count * n_features), name='dense_transform')(x)
    x = relu_bn(x)
    x = tf.reshape(x, [-1, int(f_count), n_features], name='reshape_to_features')
    return x


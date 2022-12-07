from tensorflow.keras import layers


def create_dense_encoder(x, embedding_size):
    bottleneck = layers.Dense(embedding_size)(x)
    return bottleneck


class Embedding:
    def __init__(self, embedding_size, embedding_type="flatten", activation="linear", drop_rate=0.25, mode="2d"):
        self.embedding_size = embedding_size
        self.embedding_type = embedding_type

        self.activation = activation

        self.drop_rate = drop_rate

        self.mode = mode

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
        if self.embedding_type == "None":
            latent = x
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "flatten":
            latent = layers.Flatten()(x)
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "glob_avg":
            latent = self.add_pooling_op(x, "avg")
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "glob_max":
            latent = self.add_pooling_op(x, "max")
            bottleneck = create_dense_encoder(latent, self.embedding_size)
        elif self.embedding_type == "conv":
            x = layers.Conv2D(self.embedding_size, (1, 1))(x)
            bottleneck = layers.GlobalAveragePooling2D()(x)
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

        if self.drop_rate > 0:
            bottleneck = layers.Dropout(self.drop_rate)(bottleneck)

        return bottleneck

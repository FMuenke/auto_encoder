from tensorflow.keras import layers


class Embedding:
    def __init__(self, embedding_size, embedding_type="flatten", activation="linear", drop_rate=0.25):
        self.embedding_size = embedding_size
        self.embedding_type = embedding_type

        self.activation = activation

        self.drop_rate = drop_rate

    def build(self, x):
        if self.embedding_type == "None":
            latent = x
        elif self.embedding_type == "flatten":
            latent = layers.Flatten()(x)
        elif self.embedding_type == "glob_avg":
            latent = layers.GlobalAveragePooling2D()(x)
        else:
            raise Exception("Unknown Embedding Type - {} -".format(self.embedding_type))
        if self.drop_rate > 0:
            latent = layers.Dropout(self.drop_rate)(latent)
        bottleneck = layers.Dense(self.embedding_size)(latent)

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

        return bottleneck

from tensorflow.keras import layers


class Embedding:
    def __init__(self, embedding_size, embedding_type="flatten", activation="linear", drop_rate=0.25):
        self.embedding_size = embedding_size
        self.embedding_type = embedding_type

        self.activation = activation

        self.drop_rate = drop_rate

    def create_dense_encoder(self, x):
        if self.drop_rate > 0:
            x = layers.Dropout(self.drop_rate)(x)
        bottleneck = layers.Dense(self.embedding_size)(x)
        return bottleneck

    def build(self, x):
        if self.embedding_type == "None":
            latent = x
            bottleneck = self.create_dense_encoder(latent)
        elif self.embedding_type == "flatten":
            latent = layers.Flatten()(x)
            bottleneck = self.create_dense_encoder(latent)
        elif self.embedding_type == "glob_avg":
            latent = layers.GlobalAveragePooling2D()(x)
            bottleneck = self.create_dense_encoder(latent)
        elif self.embedding_type == "glob_max":
            latent = layers.GlobalMaxPool2D()(x)
            bottleneck = self.create_dense_encoder(latent)
        elif self.embedding_type == "conv":
            if self.drop_rate > 0:
                x = layers.SpatialDropout2D(self.drop_rate)(x)
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

        return bottleneck

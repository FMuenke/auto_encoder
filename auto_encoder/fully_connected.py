from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding
from auto_encoder.sampling_layer import Sampling


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def fcn_block(x, units, ident):
    x = layers.Dense(units, name=ident)(x)
    x = relu_bn(x)
    return x


def fully_connected_auto_encoder(input_shape, embedding_size, embedding_activation, drop_rate=0.25):
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    n_inputs = int(input_shape[0] * input_shape[1] * input_shape[2])
    n_spatial = int(input_shape[0] * input_shape[1])
    x = layers.Flatten()(input_layer)

    x = fcn_block(x, n_spatial / 4, ident="enc_1")

    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type="None",
        activation=embedding_activation,
        drop_rate=drop_rate
    )

    bottleneck = emb.build(x)

    x = fcn_block(bottleneck, n_spatial / 4, ident="dec_1")

    x = layers.Dense(n_inputs, name="output")(x)
    output = layers.Reshape([int(input_shape[0]), int(input_shape[1]), input_shape[2]], name='Reshape_Layer')(x)
    return input_layer, bottleneck, output


def variational_fully_connected_auto_encoder(input_shape, embedding_size):
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    n_inputs = int(input_shape[0] * input_shape[1] * input_shape[2])
    n_spatial = int(input_shape[0] * input_shape[1])
    x = layers.Flatten()(input_layer)

    x = fcn_block(x, n_spatial / 4, ident="enc_1")

    latent_flat = layers.Flatten()(x)

    z_mean = layers.Dense(embedding_size, name="z_mean")(latent_flat)
    z_log_var = layers.Dense(embedding_size, name="z_log_var")(latent_flat)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(embedding_size,))

    x = fcn_block(latent_inputs, n_spatial / 4, ident="dec_1")

    x = layers.Dense(n_inputs, name="output")(x)
    output = layers.Reshape([int(input_shape[0]), int(input_shape[1]), input_shape[2]], name='Reshape_Layer')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")
    return encoder, decoder

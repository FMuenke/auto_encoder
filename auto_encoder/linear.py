import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding
from auto_encoder.sampling_layer import Sampling


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def make_encoder_stack(input_layer, depth, resolution):
    x = input_layer
    for i in range(depth):
        x = layers.AvgPool2D()(x)
        x = layers.SeparableConv2D(
            kernel_size=3,
            strides=1,
            filters=16 * (i + 1) * resolution,
            padding="same", name="down-{}".format(i + 1))(x)
        x = relu_bn(x)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution):
    x = feature_map_after_bottleneck
    for i in range(depth - 1):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(
            kernel_size=3,
            strides=1,
            filters=16 * (depth - i) * resolution,
            padding="same", name="up-{}".format(i + 1))(x)
        x = relu_bn(x)
    return x


def linear_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate,
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)
    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=drop_rate
    )

    bottleneck = emb.build(x)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * embedding_size * 2), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), embedding_size * 2], name='Reshape_Layer')

    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def linear_variational_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate,
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)

    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=drop_rate
    )

    latent_flat = emb.build(x)

    z_mean = layers.Dense(embedding_size, name="z_mean")(latent_flat)
    z_log_var = layers.Dense(embedding_size, name="z_log_var")(latent_flat)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(embedding_size,))

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * embedding_size * 2), name='dense_1')(latent_inputs)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), embedding_size * 2], name='Reshape_Layer')

    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.backbone.embedding import Embedding, transform_to_feature_maps
from auto_encoder.backbone.sampling_layer import Sampling

from auto_encoder.backbone.essentials import relu_bn


def make_encoder_stack(input_layer, depth, resolution):
    x = input_layer
    for i in range(depth):
        x = layers.SeparableConv2D(
            kernel_size=3,
            strides=1,
            filters=16 * 2**i * resolution,
            padding="same", name="down-{}".format(i + 1))(x)
        x = relu_bn(x)
        x = layers.AvgPool2D()(x)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution):
    x = feature_map_after_bottleneck
    for i in range(depth):
        x = layers.UpSampling2D()(x)
        x = layers.Convolution2D(
            kernel_size=3,
            strides=1,
            filters=16 * 2**(depth - i - 1) * resolution,
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
        dropout_structure,
        asymmetrical,
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
        drop_rate=drop_rate,
        dropout_structure=dropout_structure,
    )

    bottleneck = emb.build(x)
    x = bottleneck

    reshape_layer_dim = input_shape[0] / (2 ** depth) / 2
    assert reshape_layer_dim in [2 ** count for count in [0, 1, 2, 3, 4, 5, 6]]

    if asymmetrical:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_decoder_stack(x, depth, resolution=1)
    else:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def linear_variational_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        depth,
        resolution,
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)

    if embedding_type == "glob_avg":
        latent = layers.GlobalAvgPool2D()(x)
    elif embedding_type == "flatten":
        latent = layers.Flatten()(x)
    elif embedding_type == "glob_max":
        latent = layers.GlobalMaxPool2D()(x)
    else:
        raise Exception("Unknown Embedding - {} -".format(embedding_type))

    z_mean = layers.Dense(embedding_size, name="z_mean")(latent)
    z_log_var = layers.Dense(embedding_size, name="z_log_var")(latent)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(embedding_size,))

    reshape_layer_dim = input_shape[0] / (2 ** depth) / 2
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = transform_to_feature_maps(latent_inputs, reshape_layer_dim, reshape_layer_dim, embedding_size)
    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding
from auto_encoder.sampling_layer import Sampling

from auto_encoder.linear import make_decoder_stack as make_linear_decoder_stack


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def make_resnet_encoder_block(x, filters, ident, downsample=True):
    f1, f2, f3 = filters
    y = layers.Conv2D(
        kernel_size=1,
        strides=1,
        filters=f1,
        padding="same", name="r-down.1-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Conv2D(
        kernel_size=3,
        strides=(1 if not downsample else 2),
        filters=f2,
        padding="same", name="r-down.2-{}".format(ident))(y)
    y = relu_bn(y)
    y = layers.Conv2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="r-down.3-{}".format(ident))(y)

    if downsample:
        x = layers.AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = layers.Conv2D(
            kernel_size=1,
            strides=1,
            filters=f3,
            padding="same", name="r-down.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def make_resnet_decoder_block(x, filters, ident, upsample=True):
    f1, f2, f3 = filters
    y = layers.Conv2DTranspose(
        kernel_size=1,
        strides=1,
        filters=f1,
        padding="same", name="r-up.1-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Conv2DTranspose(
        kernel_size=3,
        strides=(1 if not upsample else 2),
        filters=f2,
        padding="same", name="r-up.2-{}".format(ident))(y)
    y = relu_bn(y)
    y = layers.Conv2DTranspose(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="r-up.3-{}".format(ident))(y)

    if upsample:
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2DTranspose(
            kernel_size=1,
            strides=1,
            filters=f3,
            padding="same", name="r-up.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def make_encoder_stack(input_layer, depth, resolution):
    x = input_layer
    for i in range(depth):
        filters = [
            8 * 2**i * resolution,
            8 * 2**i * resolution,
            16 * 2**i * resolution
        ]
        x = make_resnet_encoder_block(x, filters, ident=i + 1)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution):
    x = feature_map_after_bottleneck
    for i in range(depth):
        filters = [
            16 * 2**(depth - i - 1) * resolution,
            8 * 2**(depth - i - 1) * resolution,
            8 * 2**(depth - i - 1) * resolution,
        ]
        x = make_resnet_decoder_block(x, filters, ident=i + 1)
    return x


def residual_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate=0.25,
        skip=False,
        asymmetrical=False
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)

    n_features = int(x.shape[-1])

    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=(drop_rate if not skip else 0.0)
    )
    bottleneck = emb.build(x)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    if skip:
        x_pass = layers.Conv2D(embedding_size, (1, 1))(x)
        x_pass = relu_bn(x_pass)
        x_pass = layers.Flatten()(x_pass)
        x_pass = layers.Dropout(drop_rate)(x_pass)
        x = layers.Concatenate()([bottleneck, x_pass])
    else:
        x = bottleneck

    if asymmetrical:
        n_features = embedding_size
        x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * n_features), name='dense_1')(x)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), n_features], name='Reshape_Layer')
        x = make_linear_decoder_stack(x, depth, resolution=1)
    else:
        x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * n_features), name='dense_1')(x)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), n_features], name='Reshape_Layer')
        x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def residual_variational_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate=0.25
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)
    n_features = int(x.shape[-1])
    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=drop_rate
    )

    latent = emb.build(x)
    z_mean = layers.Dense(embedding_size, name="z_mean")(latent)
    z_log_var = layers.Dense(embedding_size, name="z_log_var")(latent)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(embedding_size,))

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * n_features), name='dense_1')(latent_inputs)
    x = layers.LeakyReLU()(x)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), n_features], name='Reshape_Layer')

    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


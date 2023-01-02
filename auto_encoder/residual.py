import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding, transform_to_feature_maps
from auto_encoder.sampling_layer import Sampling

from auto_encoder.linear import make_decoder_stack as make_linear_decoder_stack

from auto_encoder.essentials import relu_bn, add_classification_head


def make_clf_resnet_encoder_block(x, filters, ident, downsample=True):
    f1, f2, f3 = filters
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f1,
        padding="same", name="clf_r-down.1-{}".format(ident))(x)
    y = relu_bn(y, "1.{}".format(ident))
    y = layers.SeparableConv2D(
        kernel_size=3,
        strides=(1 if not downsample else 2),
        filters=f2,
        padding="same", name="clf_r-down.2-{}".format(ident))(y)
    y = relu_bn(y, "2.{}".format(ident))
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="clf_r-down.3-{}".format(ident))(y)

    if downsample:
        x = layers.AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = layers.SeparableConv2D(
            kernel_size=1,
            strides=1,
            filters=f3,
            padding="same", name="clf_r-down.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out, "3.{}".format(ident))
    return out


def make_resnet_encoder_block(x, filters, ident, downsample=True):
    f1, f2, f3 = filters
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f1,
        padding="same", name="r-down.1-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.SeparableConv2D(
        kernel_size=3,
        strides=(1 if not downsample else 2),
        filters=f2,
        padding="same", name="r-down.2-{}".format(ident))(y)
    y = relu_bn(y)
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="r-down.3-{}".format(ident))(y)

    if downsample:
        x = layers.AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = layers.SeparableConv2D(
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


def residual_classifier(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate,
        dropout_structure,
        noise,
        n_classes,
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
        noise=noise,
        skip=False,
    )
    x_1, _ = emb.build(x)

    filters = [
        8 * 2 ** depth * resolution,
        8 * 2 ** depth * resolution,
        16 * 2 ** depth * resolution
    ]
    x_2 = make_clf_resnet_encoder_block(x, filters, ident="final")
    x_2 = layers.GlobalAvgPool2D()(x_2)

    x = layers.Concatenate()([x_1, x_2])

    output = add_classification_head(x, n_classes=n_classes, hidden_units=[512], dropout_rate=0.75)
    return input_layer, output


def residual_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate,
        dropout_structure,
        noise,
        skip,
        asymmetrical
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
        noise=noise,
        skip=skip,
    )
    bottleneck, x = emb.build(x)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** count for count in [0, 1, 2, 3, 4, 5, 6]]

    if asymmetrical:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_linear_decoder_stack(x, depth, resolution=1)
    else:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def residual_variational_auto_encoder(
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

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** count for count in [0, 1, 2, 3, 4, 5, 6]]

    x = transform_to_feature_maps(latent_inputs, reshape_layer_dim, reshape_layer_dim, embedding_size)
    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


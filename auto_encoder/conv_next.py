import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding, transform_to_feature_maps
from auto_encoder.sampling_layer import Sampling

from auto_encoder.residual import make_decoder_stack as make_residual_decoder_stack

from auto_encoder.essentials import relu_bn, add_classification_head


def make_conv_next_encoder_block(x, filters, ident, downsample=True, mark_as_clf=False):
    f1, f2, f3 = filters

    if mark_as_clf:
        prefix = "clf_"
    else:
        prefix = ""

    if downsample:
        # x = layers.SeparableConv2D(f1, kernel_size=2, strides=2, name="{}r-down.0-{}".format(prefix, ident))(x)
        # x = relu_bn(x, "{}0.down.{}".format(prefix, ident))
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.SeparableConv2D(f1, (1, 1), name="{}r-down.0-{}".format(prefix, ident))(x)
        x = relu_bn(x, "{}0.down.{}".format(prefix, ident))

    y = layers.SeparableConv2D(
        kernel_size=7,
        strides=1,
        filters=f1,
        padding="same", name="{}r-down.1-{}".format(prefix, ident))(x)
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f2,
        padding="same", name="{}r-down.2-{}".format(prefix, ident))(y)
    y = layers.LayerNormalization()(y)
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="{}r-down.3-{}".format(prefix, ident))(y)

    y = layers.Activation(activation=tf.nn.gelu)(y)

    out = layers.Add()([x, y])
    return out


def make_conv_next_decoder_block(x, filters, ident, upsample=True):
    f1, f2, f3 = filters

    if upsample:
        x = layers.Conv2DTranspose(f1, kernel_size=3, strides=2, padding="same", name="r-up.0-{}".format(ident))(x)
        x = relu_bn(x, "0.up.norm-{}".format(ident))

    y = layers.SeparableConv2D(
        kernel_size=7,
        strides=1,
        filters=f1,
        padding="same", name="r-up.1-{}".format(ident))(x)
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f2,
        padding="same", name="r-up.2-{}".format(ident))(y)
    y = layers.LayerNormalization(name="1.up.norm-{}".format(ident))(y)
    y = layers.SeparableConv2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="r-up.3-{}".format(ident))(y)

    y = layers.Activation(activation=tf.nn.gelu)(y)

    out = layers.Add()([x, y])
    return out


def make_encoder_stack(input_layer, depth, resolution, scale=0):
    x = input_layer
    for i in range(depth):
        filters = [
            12 * 2**i * resolution,
            36 * 2**i * resolution,
            12 * 2**i * resolution
        ]
        x = make_conv_next_encoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_conv_next_encoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), downsample=False)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution, scale=0):
    x = feature_map_after_bottleneck
    for i in range(depth):
        filters = [
            12 * 2**(depth - i - 1) * resolution,
            36 * 2**(depth - i - 1) * resolution,
            12 * 2**(depth - i - 1) * resolution,
        ]
        x = make_conv_next_decoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_conv_next_decoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), upsample=False)
    return x


def conv_next_classifier_wo_embedding(
        input_shape,
        depth,
        resolution,
        scale,
        drop_rate,
        dropout_structure,
        noise,
        n_classes,
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution, scale)
    x = layers.GlobalAvgPool2D()(x)
    output = add_classification_head(x, n_classes=n_classes, hidden_units=[], dropout_rate=0.75)
    return input_layer, output


def convnext_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
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

    x = make_encoder_stack(input_layer, depth, resolution, scale=scale)

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
        x = make_residual_decoder_stack(x, depth, resolution=1)
    else:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_residual_decoder_stack(x, depth, resolution, scale=scale)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def residual_variational_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        depth,
        scale,
        resolution,
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution, scale)

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
    x = make_decoder_stack(x, depth, resolution, scale)

    output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


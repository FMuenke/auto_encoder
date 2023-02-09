import tensorflow as tf
from tensorflow.keras import layers


def add_classification_head(x, n_classes, hidden_units, dropout_rate=0.75):
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu, name="clf_{}".format(i))(x)
        x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(n_classes, name="clf_final")(x)
    return output


def add_local_dropout_2d(x, drop_rate):
    batch_size, fmh, fmw, n_f = x.shape
    return layers.Dropout(drop_rate, noise_shape=[batch_size, fmh, fmw, 1])(x)


def add_local_dropout_1d(x, drop_rate):
    batch_size, fm, n_f = x.shape
    return layers.Dropout(drop_rate, noise_shape=[batch_size, fm, 1])(x)


def add_dropout_2d(x, drop_rate, dropout_structure):
    if dropout_structure == "general":
        return layers.Dropout(drop_rate)(x)
    elif dropout_structure == "local":
        return add_local_dropout_2d(x, drop_rate)
    elif dropout_structure == "spatial":
        return layers.SpatialDropout2D(drop_rate)(x)
    else:
        raise Exception("Unknown Dropout Structure - {} -".format(dropout_structure))


def add_dropout_1d(x, drop_rate, dropout_structure):
    if dropout_structure == "general":
        return layers.Dropout(drop_rate)(x)
    elif dropout_structure == "local":
        return add_local_dropout_1d(x, drop_rate)
    elif dropout_structure == "spatial":
        return layers.SpatialDropout1D(drop_rate)(x)
    else:
        raise Exception("Unknown Dropout Structure - {} -".format(dropout_structure))


def relu_bn(inputs, name=None):
    if name is None:
        x = layers.LayerNormalization()(inputs)
        x = layers.Activation(activation=tf.nn.gelu)(x)
        return x
    else:
        x = layers.LayerNormalization(name="{}-NORM".format(name))(inputs)
        x = layers.Activation(activation=tf.nn.gelu)(x)
        return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def make_residual_encoder_block(x, filters, ident, downsample=True):
    f1, f2, f3 = filters
    y = layers.Convolution2D(
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
    y = layers.Convolution2D(
        kernel_size=1,
        strides=1,
        filters=f3,
        padding="same", name="r-down.3-{}".format(ident))(y)

    if downsample:
        x = layers.AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = layers.Convolution2D(
            kernel_size=1,
            strides=1,
            filters=f3,
            padding="same", name="r-down.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def make_small_residual_encoder_block(x, filters, ident, downsample=True):
    f1, f2 = filters
    y = layers.SeparableConv2D(
        kernel_size=3,
        strides=(1 if not downsample else 2),
        filters=f1,
        padding="same", name="r-down.2-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Convolution2D(
        kernel_size=3,
        strides=1,
        filters=f2,
        padding="same", name="r-down.3-{}".format(ident))(y)

    if downsample:
        x = layers.SeparableConv2D(
            kernel_size=1,
            strides=2,
            filters=f2,
            padding="same", name="r-down.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def make_residual_decoder_block(x, filters, ident, upsample=True):
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


def make_small_residual_decoder_block(x, filters, ident, upsample=True):
    f1, f2 = filters
    y = layers.Conv2DTranspose(
        kernel_size=3,
        strides=(1 if not upsample else 2),
        filters=f1,
        padding="same", name="r-up.2-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Convolution2D(
        kernel_size=1,
        strides=1,
        filters=f2,
        padding="same", name="r-up.3-{}".format(ident))(y)

    if upsample:
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Convolution2D(
            kernel_size=1,
            strides=1,
            filters=f2,
            padding="same", name="r-up.pass-{}".format(ident))(x)

    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out

import tensorflow as tf
from tensorflow.keras import layers


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


def resnet_auto_encoder(input_shape, embedding_size, depth, resolution, drop_rate=0.25):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = input_layer

    for i in range(depth):
        filters = [
            16 * (i + 1) * resolution,
            16 * (i + 1) * resolution,
            32 * (i + 1) * resolution
        ]
        x = make_resnet_encoder_block(x, filters, ident=i + 1)

    latent_flat = layers.Flatten()(x)
    latent_flat = layers.Dropout(drop_rate)(latent_flat)
    bottleneck = layers.Dense(embedding_size)(latent_flat)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 64), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 64], name='Reshape_Layer')

    for i in range(depth - 1):
        filters = [
            16 * (depth - i) * resolution,
            16 * (depth - i) * resolution,
            8 * (depth - i) * resolution,
        ]
        x = make_resnet_decoder_block(x, filters, ident=i + 1)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def resnet50_auto_encoder(input_shape, embedding_size, drop_rate=0.25):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    depth = 4
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    x = input_layer

    x = make_resnet_encoder_block(x, [16, 16, 64], ident="1-a", downsample=True)
    x = make_resnet_encoder_block(x, [16, 16, 64], ident="1-b", downsample=False)
    x = make_resnet_encoder_block(x, [16, 16, 64], ident="1-c", downsample=False)

    x = make_resnet_encoder_block(x, [32, 32, 128], ident="2-a", downsample=True)
    x = make_resnet_encoder_block(x, [32, 32, 128], ident="2-b", downsample=False)
    x = make_resnet_encoder_block(x, [32, 32, 128], ident="2-c", downsample=False)
    x = make_resnet_encoder_block(x, [32, 32, 128], ident="2-d", downsample=False)

    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-a", downsample=True)
    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-b", downsample=False)
    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-c", downsample=False)
    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-d", downsample=False)
    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-e", downsample=False)
    x = make_resnet_encoder_block(x, [64, 64, 256], ident="3-f", downsample=False)

    x = make_resnet_encoder_block(x, [128, 128, 512], ident="4-a", downsample=True)
    x = make_resnet_encoder_block(x, [128, 128, 512], ident="4-b", downsample=False)
    x = make_resnet_encoder_block(x, [128, 128, 512], ident="4-c", downsample=False)

    latent_flat = layers.Flatten()(x)
    latent_flat = layers.Dropout(drop_rate)(latent_flat)
    bottleneck = layers.Dense(embedding_size)(latent_flat)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 512), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 512], name='Reshape_Layer')

    x = make_resnet_decoder_block(x, 128 * resolution, ident="4-c", upsample=False)
    x = make_resnet_decoder_block(x, 128 * resolution, ident="4-b", upsample=False)
    x = make_resnet_decoder_block(x, 64 * resolution, ident="4-a", upsample=True)

    x = make_resnet_decoder_block(x, 64 * resolution, ident="3-f", upsample=False)
    x = make_resnet_decoder_block(x, 64 * resolution, ident="3-e", upsample=False)
    x = make_resnet_decoder_block(x, 64 * resolution, ident="3-d", upsample=False)
    x = make_resnet_decoder_block(x, 64 * resolution, ident="3-c", upsample=False)
    x = make_resnet_decoder_block(x, 64 * resolution, ident="3-b", upsample=False)
    x = make_resnet_decoder_block(x, 32 * resolution, ident="3-a", upsample=True)

    x = make_resnet_decoder_block(x, 32 * resolution, ident="2-d", upsample=False)
    x = make_resnet_decoder_block(x, 32 * resolution, ident="2-c", upsample=False)
    x = make_resnet_decoder_block(x, 32 * resolution, ident="2-b", upsample=False)
    x = make_resnet_decoder_block(x, 16 * resolution, ident="2-a", upsample=True)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output
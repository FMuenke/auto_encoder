from tensorflow.keras import applications
import tensorflow as tf
from tensorflow.keras import layers


def make_decoder_block(x, filters, ident):
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name='conv_transpose_{}'.format(ident))(x)
    x = layers.BatchNormalization(name='bn_d{}'.format(ident))(x)
    x = layers.LeakyReLU(name='lrelu_d{}'.format(ident))(x)
    return x


def make_encoder_block(x, filters, ident):
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', name='conv_{}'.format(ident))(x)
    x = layers.BatchNormalization(name='bn_{}'.format(ident))(x)
    x = layers.LeakyReLU(name="lrelu_{}".format(ident))(x)
    return x


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def make_resnet_encoder_block(x, filters, ident, downsample=True):
    y = layers.Conv2D(
        kernel_size=3,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding="same", name="r-down.1-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Conv2D(
        kernel_size=3,
        strides=1,
        filters=filters,
        padding="same", name="r-down.2-{}".format(ident))(y)

    if downsample:
        x = layers.Conv2D(
            kernel_size=1,
            strides=2,
            filters=filters,
            padding="same", name="r-down.3-{}".format(ident))(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def make_resnet_decoder_block(x, filters, ident, upsample=True):
    y = layers.Conv2DTranspose(
        kernel_size=3,
        strides=(1 if not upsample else 2),
        filters=filters,
        padding="same", name="r-up.1-{}".format(ident))(x)
    y = relu_bn(y)
    y = layers.Conv2D(
        kernel_size=3,
        strides=1,
        filters=filters,
        padding="same", name="r-up.2-{}".format(ident))(y)

    if upsample:
        x = layers.Conv2DTranspose(
            kernel_size=1,
            strides=2,
            filters=filters,
            padding="same", name="r-up.3-{}".format(ident))(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def resnet_auto_encoder(input_shape, embedding_size, depth, resolution):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = input_layer

    for i in range(depth):
        x = make_resnet_encoder_block(x, 16 * (i + 1) * resolution, ident=i + 1)

    latent_flat = layers.Flatten()(x)
    bottleneck = layers.Dense(embedding_size)(latent_flat)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 64), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 64], name='Reshape_Layer')

    for i in range(depth - 1):
        x = make_resnet_decoder_block(x, 16 * (depth - i) * resolution, ident=i + 1)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='hard_sigmoid', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def resnet50_auto_encoder(input_shape, embedding_size, resolution=1):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    depth = 4
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    x = input_layer

    x = make_resnet_encoder_block(x, 16 * resolution, ident="1-a", downsample=True)
    x = make_resnet_encoder_block(x, 16 * resolution, ident="1-b", downsample=False)
    x = make_resnet_encoder_block(x, 16 * resolution, ident="1-c", downsample=False)

    x = make_resnet_encoder_block(x, 32 * resolution, ident="2-a", downsample=True)
    x = make_resnet_encoder_block(x, 32 * resolution, ident="2-b", downsample=False)
    x = make_resnet_encoder_block(x, 32 * resolution, ident="2-c", downsample=False)
    x = make_resnet_encoder_block(x, 32 * resolution, ident="2-d", downsample=False)

    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-a", downsample=True)
    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-b", downsample=False)
    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-c", downsample=False)
    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-d", downsample=False)
    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-e", downsample=False)
    x = make_resnet_encoder_block(x, 64 * resolution, ident="3-f", downsample=False)

    x = make_resnet_encoder_block(x, 128 * resolution, ident="4-a", downsample=True)
    x = make_resnet_encoder_block(x, 128 * resolution, ident="4-b", downsample=False)
    x = make_resnet_encoder_block(x, 128 * resolution, ident="4-c", downsample=False)

    latent_flat = layers.Flatten()(x)
    bottleneck = layers.Dense(embedding_size)(latent_flat)

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 128), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 128], name='Reshape_Layer')

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

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='hard_sigmoid', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output


def basic_auto_encoder(input_shape, embedding_size, depth, resolution):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = input_layer

    for i in range(depth):
        x = make_encoder_block(x, 16*(i+1)*resolution, ident=i+1)

    latent_flat = layers.Flatten()(x)
    bottleneck = layers.Dense(embedding_size)(latent_flat)

    reshape_layer_dim = input_shape[0] / (2**depth)
    assert reshape_layer_dim in [2**x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 64), name='dense_1')(bottleneck)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 64], name='Reshape_Layer')

    for i in range(depth - 1):
        x = make_decoder_block(x, 16 * (depth - i) * resolution, ident=i+1)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='hard_sigmoid', name='conv_transpose_5')(x)

    return input_layer, bottleneck, output


class Backbone:
    def __init__(self, backbone_type, embedding_size, depth, resolution, loss_type="mse", weights="imagenet"):
        self.backbone_type = backbone_type
        self.embedding_size = embedding_size

        self.depth = depth
        self.resolution = resolution

        self.loss_type = loss_type
        self.weights = weights

    def loss(self):
        if self.loss_type in ["mean_squared_error", "mse"]:
            return "mean_squared_error"

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def metric(self):
        return ["mse"]

    def build(self, input_shape, add_decoder=True):
        if self.backbone_type in ["resnet"]:
            x_input, bottleneck, output = resnet_auto_encoder(
                input_shape=input_shape,
                embedding_size=self.embedding_size,
                depth=self.depth,
                resolution=self.resolution
            )
        elif self.backbone_type in ["resnet50"]:
            x_input, bottleneck, output = resnet50_auto_encoder(
                input_shape=input_shape,
                embedding_size=self.embedding_size,
                resolution=self.resolution
            )
        elif self.backbone_type in ["basic"]:
            x_input, bottleneck, output = basic_auto_encoder(
                input_shape=input_shape,
                embedding_size=self.embedding_size,
                depth=self.depth,
                resolution=self.resolution
            )
        else:
            raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

        if add_decoder:
            return x_input, output
        else:
            return x_input, bottleneck


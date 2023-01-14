import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.essentials import relu_bn, add_dropout_2d, make_residual_encoder_block
from auto_encoder.residual import make_encoder_stack, make_decoder_stack
from auto_encoder.linear import make_decoder_stack as make_linear_decoder_stack
from auto_encoder.embedding import Embedding, transform_to_feature_maps


def still_auto_encoder(
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
        asymmetrical,
        patch_size=4
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = layers.Convolution2D(8 * resolution, (patch_size, patch_size), (patch_size, patch_size))(input_layer)
    x = relu_bn(x)

    x = add_dropout_2d(x, drop_rate, dropout_structure)

    x = make_encoder_stack(x, depth, resolution)

    x = make_residual_encoder_block(x, [
        int(0.5 * embedding_size), int(0.5 * embedding_size), embedding_size
    ], ident="EMBEDDING", downsample=True)

    bottleneck = layers.GlobalAvgPool2D()(x)

    x = make_decoder_stack(x, depth, resolution=resolution)

    output = layers.Conv2DTranspose(3, 1, 1, padding='same', activation='linear', name='final')(x)
    return input_layer, bottleneck, output

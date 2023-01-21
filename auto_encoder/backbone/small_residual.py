from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.backbone.embedding import Embedding, transform_to_feature_maps
from auto_encoder.backbone.sampling_layer import Sampling

from auto_encoder.backbone.essentials import make_small_residual_encoder_block
from auto_encoder.backbone.essentials import make_small_residual_decoder_block


def make_encoder_stack(input_layer, depth, resolution, scale=0):
    x = input_layer
    for i in range(depth):
        filters = [
            8 * 2**i * resolution,
            8 * 2**i * resolution,
        ]
        x = make_small_residual_encoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_small_residual_encoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), downsample=False)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution, scale=0):
    x = feature_map_after_bottleneck
    for i in range(depth):
        filters = [
            8 * 2**(depth - i - 1) * resolution,
            8 * 2**(depth - i - 1) * resolution,
        ]
        x = make_small_residual_decoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_small_residual_decoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), upsample=False)
    return x


def small_residual_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
        resolution,
        drop_rate,
        dropout_structure,
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
        x = make_decoder_stack(x, depth, resolution, scale=scale)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return input_layer, bottleneck, output

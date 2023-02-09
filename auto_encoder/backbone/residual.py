from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.backbone.embedding import Embedding, transform_to_feature_maps
from auto_encoder.backbone.sampling_layer import Sampling

from auto_encoder.backbone.essentials import make_residual_encoder_block, make_residual_decoder_block
from auto_encoder.backbone.essentials import relu_bn, add_dropout_2d


def make_encoder_stack(input_layer, depth, resolution, scale=0):
    x = input_layer
    for i in range(depth):
        filters = [
            8 * 2**i * resolution,
            8 * 2**i * resolution,
            16 * 2**i * resolution
        ]
        x = make_residual_encoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_residual_encoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), downsample=False)
    return x


def make_decoder_stack(feature_map_after_bottleneck, depth, resolution, scale=0):
    x = feature_map_after_bottleneck
    for i in range(depth):
        filters = [
            16 * 2**(depth - i - 1) * resolution,
            8 * 2**(depth - i - 1) * resolution,
            8 * 2**(depth - i - 1) * resolution,
        ]
        x = make_residual_decoder_block(x, filters, ident=i + 1)
        if scale > 0:
            for scale in range(scale):
                x = make_residual_decoder_block(x, filters, ident="{}-ident-{}".format(i + 1, scale), upsample=False)
    return x


def make_residual_decoder(x, input_shape, depth, resolution, scale, embedding_size, asymmetrical):
    reshape_layer_dim = input_shape[0] / (2 ** depth) / 2
    assert reshape_layer_dim in [2 ** count for count in [0, 1, 2, 3, 4, 5, 6]]

    if asymmetrical:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_decoder_stack(x, depth, resolution=1)
    else:
        x = transform_to_feature_maps(x, reshape_layer_dim, reshape_layer_dim, embedding_size)
        x = make_decoder_stack(x, depth, resolution, scale=scale)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    return output


def make_residual_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
        resolution,
        drop_rate,
        dropout_structure,
        use_stem=False,
        p_size=0
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    if use_stem:
        x = layers.Convolution2D(8 * resolution, (7, 7), (3, 3), name="STEM")(input_layer)
        x = relu_bn(x, name="STEM-NORM")
        x = layers.MaxPool2D()(x)
    elif p_size > 0:
        patch_filters = p_size * p_size
        x = layers.Convolution2D(patch_filters, (p_size, p_size), (p_size, p_size), name="PATCHIFY")(input_layer)
        x = relu_bn(x, name="PATCHIFY_NORM")
        if dropout_structure.startswith("patch_"):
            print("[INFO] PATCH DROPOUT ACTIVE")
            dropout_structure = dropout_structure.replace("patch_", "")
            x = add_dropout_2d(x, drop_rate, dropout_structure)
            # REMOVE DROPOUT FOR EMBEDDING
            drop_rate = 0
            dropout_structure = ""
    else:
        x = input_layer
    x = make_encoder_stack(x, depth, resolution, scale=scale)

    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=drop_rate,
        dropout_structure=dropout_structure,
    )
    bottleneck = emb.build(x)
    return input_layer, bottleneck


def residual_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
        resolution,
        drop_rate,
        dropout_structure,
        asymmetrical,
        use_stem=False,
        patch_size=0,
):
    input_layer, bottleneck = make_residual_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
        resolution,
        drop_rate,
        dropout_structure,
        use_stem=use_stem,
        p_size=patch_size,
    )
    x = bottleneck
    output = make_residual_decoder(x, input_shape, depth, resolution, scale, embedding_size, asymmetrical)
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
    x = latent_inputs
    output = make_residual_decoder(x, input_shape, depth, resolution, scale, embedding_size, asymmetrical)
    decoder = keras.Model(latent_inputs, output, name="decoder")

    return encoder, decoder


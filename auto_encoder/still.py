import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.essentials import relu_bn, add_dropout_2d
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
        asymmetrical
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    x = make_encoder_stack(input_layer, depth, resolution)

    batch_size, f_m_h, f_m_w, n_features = x.shape

    latent = layers.Conv2D(embedding_size, (1, 1), strides=1, padding="same")(x)
    relu_bn(latent)
    latent = add_dropout_2d(latent, drop_rate, dropout_structure)
    bottleneck = layers.GlobalAvgPool2D()(latent)
    x = layers.Flatten()(latent)
    x = layers.Dense(embedding_size)(x)

    x = transform_to_feature_maps(x, f_m_h, f_m_w, embedding_size)
    x = make_decoder_stack(x, depth, resolution)

    output = layers.Conv2DTranspose(3, 1, 1, padding='same', activation='linear', name='final')(x)
    return input_layer, bottleneck, output

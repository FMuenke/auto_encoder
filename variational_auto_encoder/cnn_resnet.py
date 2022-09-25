import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.cnn_resnet import make_resnet_encoder_block, make_resnet_decoder_block
from variational_auto_encoder.sampling_layer import Sampling


def residual_auto_encoder(input_shape, embedding_size, depth, resolution, drop_rate=0.25):
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

    z_mean = layers.Dense(embedding_size, name="z_mean")(latent_flat)
    z_log_var = layers.Dense(embedding_size, name="z_log_var")(latent_flat)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    # latent_inputs = keras.Input(shape=(embedding_size,))

    reshape_layer_dim = input_shape[0] / (2 ** depth)
    assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

    x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * 64), name='dense_1')(z)
    x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), 64], name='Reshape_Layer')

    for i in range(depth - 1):
        filters = [
            16 * (depth - i) * resolution,
            16 * (depth - i) * resolution,
            8 * (depth - i) * resolution,
        ]
        x = make_resnet_decoder_block(x, filters, ident=i + 1)

    output = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='linear', name='conv_transpose_5')(x)
    decoder = keras.Model(input_layer, output, name="decoder")

    return encoder, decoder

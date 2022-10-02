import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from auto_encoder.embedding import Embedding


def asymmetric_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        drop_rate=0.25
):
    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])

    base_model = keras.applications.xception.Xception(
        include_top=False,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
    )
    input_layer = base_model.input
    layer_output = base_model.output

    emb = Embedding(
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        activation=embedding_activation,
        drop_rate=drop_rate
    )

    bottleneck = emb.build(layer_output)

    height = input_shape[0] / 2
    width = input_shape[1] / 2
    channel = input_shape[2]

    x = layers.Dense(int(height * width * channel), name='direct_decode')(bottleneck)
    x = tf.reshape(x, [-1, int(height), int(width), channel], name='Reshape_Layer')
    x = layers.UpSampling2D()(x)

    output = x
    return input_layer, bottleneck, output

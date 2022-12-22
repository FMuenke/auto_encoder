import tensorflow as tf
from tensorflow.keras import layers


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



def relu_bn(inputs):
    x = layers.LayerNormalization()(inputs)
    x = layers.Activation(activation=tf.nn.gelu)(x)
    return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
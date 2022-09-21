from tensorflow.keras import layers


def relu_bn(inputs):
    relu = layers.LeakyReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def fcn_block(x, units, ident):
    x = layers.Dense(units, name=ident)(x)
    x = relu_bn(x)
    return x


def fully_connected_auto_encoder(input_shape, embedding_size, depth, resolution):
    input_layer = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    n_inputs = int(input_shape[0] * input_shape[1] * input_shape[2])
    n_spatial = int(input_shape[0] * input_shape[1])
    x = layers.Flatten()(input_layer)

    x = fcn_block(x, n_spatial / 4, ident="enc_1")

    bottleneck = layers.Dense(embedding_size, name="bootleneck")(x)
    bottleneck = relu_bn(bottleneck)

    x = fcn_block(bottleneck, n_spatial / 4, ident="dec_1")

    x = layers.Dense(n_inputs, name="output")(x)
    output = layers.Reshape([int(input_shape[0]), int(input_shape[1]), input_shape[2]], name='Reshape_Layer')(x)
    return input_layer, bottleneck, output

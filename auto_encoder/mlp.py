import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from auto_encoder.embedding import Embedding
from auto_encoder.linear import make_decoder_stack


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


def mlp_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate=0.25,
        num_patches_p_side=8,
        positional_encoding=False,
        asymmetrical=False,
):
    num_patches = num_patches_p_side**2

    embedding_dim = 16 * resolution
    model_dropout_rate = 0.2

    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    patch_size = input_shape[0] // num_patches_p_side

    inputs = layers.Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    encoder_blocks = keras.Sequential(
        [MLPMixerLayer(num_patches, embedding_dim, model_dropout_rate) for _ in range(depth)]
    )
    patches = Patches(patch_size, num_patches)(inputs)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
        x = x + position_embedding
    # Process x using the module blocks.
    x = encoder_blocks(x)

    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    emb = Embedding(
        embedding_type=embedding_type,
        embedding_size=embedding_size,
        activation=embedding_activation,
        drop_rate=drop_rate,
        mode="1d"
    )
    bottleneck = emb.build(representation)

    if asymmetrical:
        reshape_layer_dim = input_shape[0] / (2 ** depth)
        assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

        x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * embedding_size))(bottleneck)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), embedding_size], name='Reshape_Layer')
        x = make_decoder_stack(x, depth, resolution=1)
        output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_final')(x)
    else:
        decoder_blocks = keras.Sequential(
            [MLPMixerLayer(num_patches, embedding_dim, model_dropout_rate) for _ in range(depth)]
        )
        x = layers.Dense(int(num_patches * embedding_dim))(bottleneck)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(num_patches), embedding_dim], name='Reshape_Layer')
        x = decoder_blocks(x)
        x = layers.Flatten()(x)
        pre_final = layers.Dense(units=input_shape[0] * input_shape[1] * 3, activation="linear")(x)
        output = layers.Reshape((input_shape[0], input_shape[1], 3))(pre_final)

    return inputs, bottleneck, output



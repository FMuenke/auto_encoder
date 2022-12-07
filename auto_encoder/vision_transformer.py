import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


from auto_encoder.embedding import Embedding
from auto_encoder.linear import make_decoder_stack


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

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
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def transformer_layers(encoded_patches, depth, transformer_units, projection_dim, num_heads=4):
    # Create multiple layers of the Transformer block.
    for _ in range(depth):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    return encoded_patches


def vit_auto_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        resolution,
        drop_rate=0.25,
        asymmetrical=False
):
    num_patches_p_side = 8
    num_patches = num_patches_p_side**2
    num_heads = 4
    projection_dim = 16 * resolution
    transformer_units = [projection_dim * 2, projection_dim]

    input_sizes = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0, }
    assert input_shape[0] == input_shape[1], "Only Squared Inputs! - {} / {} -".format(input_shape[0], input_shape[1])
    assert input_shape[0] in input_sizes, "Input Size is not supported ({})".format(input_shape[0])
    patch_size = input_shape[0] // num_patches_p_side

    input_layer = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(input_layer)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    encoded_patches = transformer_layers(
        encoded_patches,
        depth,
        transformer_units,
        projection_dim,
        num_heads
    )

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    emb = Embedding(
        embedding_type=embedding_type,
        embedding_size=embedding_size,
        activation=embedding_activation,
        drop_rate=drop_rate,
        mode="1d"
    )
    bottleneck = emb.build(representation)

    if asymmetrical:
        print("[INFO] Asymmetrical Decoder")
        reshape_layer_dim = input_shape[0] / (2 ** depth)
        assert reshape_layer_dim in [2 ** x for x in [0, 1, 2, 3, 4, 5, 6]]

        x = layers.Dense(int(reshape_layer_dim * reshape_layer_dim * embedding_size))(bottleneck)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(reshape_layer_dim), int(reshape_layer_dim), embedding_size], name='Reshape_Layer')
        x = make_decoder_stack(x, depth, resolution=1)
        output = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='linear', name='conv_transpose_final')(x)
    else:
        x = layers.Dense(int(num_patches * projection_dim))(bottleneck)
        x = layers.LeakyReLU()(x)
        x = tf.reshape(x, [-1, int(num_patches), projection_dim], name='Reshape_Layer')
        x = transformer_layers(
            x,
            depth,
            transformer_units,
            projection_dim,
            num_heads
        )
        x = layers.Flatten()(x)
        pre_final = layers.Dense(units=input_shape[0] * input_shape[1] * 3, activation="linear")(x)
        output = layers.Reshape((input_shape[0], input_shape[1], 3))(pre_final)

    return input_layer, bottleneck, output

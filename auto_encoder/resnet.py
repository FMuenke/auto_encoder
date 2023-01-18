import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, ReLU, Add
from tensorflow.keras.layers import AveragePooling2D

from auto_encoder.embedding import Embedding


def stem(inputs):
    ''' Construct Stem Convolutional Group
        inputs : the input vector
    '''
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def learner(x, n_blocks):
    """ Construct the Learner
        x          : input to the learner
        n_blocks   : number of blocks in a group
    """
    # First Residual Block Group of 16 filters (Stage 1)
    # Quadruple (4X) the size of filters to fit the next Residual Group
    x = residual_group(x, 16, n_blocks, strides=(1, 1), n=4)

    # Second Residual Block Group of 64 filters (Stage 2)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 64, n_blocks, n=2)

    # Third Residual Block Group of 64 filters (Stage 3)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 128, n_blocks, n=2)
    return x


def residual_group(x, n_filters, n_blocks, strides=(2, 2), n=2):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
        strides   : whether the projection block is a strided convolution
        n         : multiplier for the number of filters out
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides, n=n)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters, n)
    return x


def identity_block(x, n_filters, n=2):
    """ Construct a Bottleneck Residual Block of Convolutions
        x        : input into the block
        n_filters: number of filters
        n        : multiplier for filters out
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of output filters by 2X or 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([x, shortcut])
    return x


def projection_block(x, n_filters, strides=(2, 2), n=2):
    """ Construct a Bottleneck Residual Block with Projection Shortcut
        Increase the number of filters by 2X (or 4X on first stage)
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
        n        : multiplier for number of filters out
    """
    # Construct the projection shortcut
    # Increase filters by 2X (or 4X) to match shape when added to output of block
    shortcut = Conv2D(n_filters * n, (1, 1), strides=strides, kernel_initializer='he_normal')(x)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    # Bottleneck layer - feature pooling when strides=(2, 2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of filters by 2X (or 4X)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([shortcut, x])
    return x


def classifier(x, n_classes):
    ''' Construct a Classifier
        x         : input into the classifier
        n_classes : number of classes
    '''
    # Pool the feature maps after the end of all the residual blocks
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=8)(x)

    # Flatten into 1D vector
    x = Flatten()(x)

    # Final Dense Outputting Layer
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs


# -------------------
# Model      | n   |
# ResNet20   | 2   |
# ResNet56   | 6   |
# ResNet110  | 12  |
# ResNet164  | 18  |
# ResNet1001 | 111 |
#

def resnet_encoder(
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        n_blocks,
        drop_rate,
        dropout_structure
):

    print("[INFO] RESNET {}".format(n_blocks))
    depth = n_blocks * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    input_layer = Input(batch_shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    # The Stem Convolution Group
    x = stem(input_layer)

    # The learner
    x = learner(x, n_blocks)

    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    WEIGHT_DECAY = 0.0005

    # Projection head.
    x = layers.Dense(
        embedding_size, use_bias=False, kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        embedding_size, use_bias=False, kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    encoder = keras.Model(input_layer, x, name="encoder")

    predictor = keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            layers.Input((embedding_size,)),
            layers.Dense(
                int(embedding_size / 4),
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(0.0005),
            ),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dense(embedding_size),
        ],
        name="predictor",
    )

    return encoder, predictor
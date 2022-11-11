import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error


def entropy_minimum(n_classes):
    def entropy_minimum_func(y_true, y_pred):
        idx = K.argmax(y_pred, axis=1)
        y_true = K.one_hot(idx, n_classes)
        value = categorical_crossentropy(y_true, y_pred)
        return tf.reduce_sum(value)

    return entropy_minimum_func


def mean_sq_err():
    def mean_sq_err_func(y_true, y_pred):
        value = mean_squared_error(y_true, y_pred)
        return tf.reduce_mean(value)
    return mean_sq_err_func


def dice():
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    return dice_coef_loss


def jaccard():
    def jaccard_distance_loss(y_true, y_pred, smooth=100):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is useful for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disappearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    return jaccard_distance_loss


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss_temp = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss_temp)

    return loss


def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss_temp = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss_temp)

    return loss


def mixed():
    def loss(y_true, y_pred):
        def dice_loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
            denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

            return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


if __name__ == "__main__":
    loss_entropy = entropy_minimum(100)
    y_true = K.zeros((8, 100), dtype=tf.float64)
    y_pred = K.one_hot([10, 20, 30, 40, 50, 60, 70], 100)
    l_val = loss_entropy(y_true, y_pred)
    print(l_val)

    data_1 = np.random.randint(255, size=(4, 128, 128, 3)) / 255
    data_2 = np.random.randint(255, size=(4, 128, 128, 3)) / 255
    loss_mse = mean_sq_err()

    l_val_2 = loss_mse(data_1, data_2)
    print(l_val_2)

    print(l_val + l_val_2)

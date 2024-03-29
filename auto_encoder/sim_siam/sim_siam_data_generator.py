import os.path

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

from auto_encoder.util import prepare_input_sim_clr
from auto_encoder.augmentations import Augmentations, apply_crop


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def flip_random_crop(image, image_size):
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (image_size, image_size, 3))
    return image


def custom_augment(image, image_size):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image, image_size)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        tag_set,
        batch_size,
        image_size,
        augmentations=None,
        shuffle=True,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.tag_set = tag_set
        self.shuffle = shuffle
        self.crop_strength = 0.75
        self.augmentations = augmentations
        self.baseline_augmentations = Augmentations(
            brightness=0.60,
            channel_shift=0.20,
            color_drop=0.20,
            flip_rotate90=0.50,
        )
        self.indexes = np.arange(len(self.tag_set))
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.tag_set) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch of data.
        Args:
            index (int)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        tags_temp = [self.tag_set[k] for k in indexes]

        x, y = self.__data_generation(tags_temp)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tags_temp):
        """
        Generates data containing the batch_size samples.
        """
        x = []
        y = []
        for i, tag in enumerate(tags_temp):
            img_1 = tag.load_x()
            img_2 = np.copy(img_1)

            dummy_1 = np.zeros(img_1.shape)
            dummy_2 = np.zeros(img_2.shape)

            img_1, _ = apply_crop(img_1, dummy_1, percentage=np.random.randint(100 * self.crop_strength) / 100)
            img_2, _ = apply_crop(img_2, dummy_2, percentage=np.random.randint(100 * self.crop_strength) / 100)

            img_1, _ = self.baseline_augmentations.apply(img_1, dummy_1)
            img_2, _ = self.baseline_augmentations.apply(img_2, dummy_2)

            if self.augmentations is not None:
                img_1, _ = self.augmentations.apply(img_1, dummy_1)
                img_2, _ = self.augmentations.apply(img_2, dummy_2)

            img_1 = prepare_input_sim_clr(img_1, self.image_size)
            img_2 = prepare_input_sim_clr(img_2, self.image_size)
            x.append(img_1)
            y.append(img_2)

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return x, y

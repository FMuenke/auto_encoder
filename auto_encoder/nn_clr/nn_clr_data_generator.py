from tensorflow import keras
import numpy as np
import cv2


from auto_encoder.util import prepare_input_sim_clr
from auto_encoder.augmentations import Augmentations, apply_crop


class NNCLRDataGenerator(keras.utils.Sequence):
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

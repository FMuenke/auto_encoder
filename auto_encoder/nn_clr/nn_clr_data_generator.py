from tensorflow import keras
import numpy as np
import cv2


from auto_encoder.util import prepare_input_sim_clr
from auto_encoder.augmentations import Augmentations


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
        self.augmentations = Augmentations(
            neutral_percentage=0.0,
            masking=0.0,
            cross_cut=0.0,
            patch_rotation=0.0,
            patch_shuffling=0.0,
            blurring=0.0,
            noise=0.0,
            flip_rotate90=1.0,
            crop=0.20,
            patch_masking=0.0,
            warp=0.0
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

            img_1, _ = self.augmentations.apply(img_1, img_1)
            img_2, _ = self.augmentations.apply(img_2, img_2)

            img_1 = prepare_input_sim_clr(img_1, self.image_size)
            img_2 = prepare_input_sim_clr(img_2, self.image_size)
            x.append(img_1)
            y.append(img_2)

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return x, y

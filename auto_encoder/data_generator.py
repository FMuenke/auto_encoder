from tensorflow import keras
import numpy as np

from auto_encoder.util import prepare_input


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
        self.augmentations = augmentations
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
            img = tag.load_x()
            tar = img

            if self.augmentations is not None:
                img, tar = self.augmentations.apply(img, tar)

            img = prepare_input(img, self.image_size)
            tar = prepare_input(tar, self.image_size)
            x.append(img)
            y.append(tar)

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return x, y


class ClassificationDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        tag_set,
        batch_size,
        image_size,
        class_mapping,
        augmentations=None,
        shuffle=True,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.tag_set = tag_set
        self.class_mapping = class_mapping
        self.shuffle = shuffle
        self.augmentations = augmentations
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
        y1 = []
        y2 = []
        for i, tag in enumerate(tags_temp):
            img = tag.load_x()
            tar = img
            cls = tag.load_y()

            if self.augmentations is not None:
                img, tar = self.augmentations.apply(img, tar)

            img = prepare_input(img, self.image_size)
            tar = prepare_input(tar, self.image_size)
            x.append(img)
            cls_vec = np.zeros(len(self.class_mapping))
            cls_vec[self.class_mapping[cls]] = 1
            y1.append(cls_vec)
            y2.append(tar)

        x = np.array(x, dtype=np.float32)
        y1 = np.array(y1, dtype=np.float32)
        y2 = np.array(y2, dtype=np.float32)
        return x, y1


class HybridDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        tag_set,
        batch_size,
        image_size,
        class_mapping,
        augmentations=None,
        shuffle=True,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.tag_set = tag_set
        self.class_mapping = class_mapping
        self.shuffle = shuffle
        self.augmentations = augmentations
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
        y1 = []
        y2 = []
        for i, tag in enumerate(tags_temp):
            img = tag.load_x()
            tar = img
            cls = tag.load_y()

            if self.augmentations is not None:
                img, tar = self.augmentations.apply(img, tar)

            img = prepare_input(img, self.image_size)
            tar = prepare_input(tar, self.image_size)
            x.append(img)
            cls_vec = np.zeros(len(self.class_mapping))
            cls_vec[self.class_mapping[cls]] = 1
            y1.append(cls_vec)
            y2.append(tar)

        x = np.array(x, dtype=np.float32)
        y1 = np.array(y1, dtype=np.float32)
        y2 = np.array(y2, dtype=np.float32)
        return x, [y1, y2]

import os
import cv2


class TrainImage:
    def __init__(self, base_path, image_name):
        self.path = os.path.join(base_path, image_name)
        self.name = image_name

    def load_x(self):
        return cv2.imread(self.path)


def load_folder(path_to_folder):
    images = []
    for f in os.listdir(path_to_folder):
        if f.endswith((".png", ".jpg", ".jpeg", ".ppm")):
            images.append((TrainImage(path_to_folder, f)))

        if os.path.isdir(os.path.join(path_to_folder, f)):
            sub_images = load_folder(os.path.join(path_to_folder, f))
            images += sub_images

    return images


class DataSet:
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.images = []

    def load(self):
        self.images = load_folder(self.path)
        print("Found: {} Images".format(len(self.images)))

    def get_data(self, split=0.0):
        if split == 0:
            return self.images
        else:
            assert 0 < split < 1.0, "Invalid Split {}".format(split)
            return self.images[:int(len(self.images)*split)], self.images[int(len(self.images)*split):]

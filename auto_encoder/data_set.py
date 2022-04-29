import os
import cv2


class TrainImage:
    def __init__(self, base_path, image_name):
        self.path = os.path.join(base_path, image_name)
        self.name = image_name

    def load_x(self):
        return cv2.imread(self.path)


class DataSet:
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.images = []

    def load(self):
        for f in os.listdir(self.path):
            if f.endswith((".png", ".jpg", ".jpeg", ".ppm")):
                self.images.append(TrainImage(self.path, f))

        print("Found: {} Images".format(len(self.images)))

    def get_data(self, split=0.0):
        if split == 0:
            return self.images
        else:
            assert 0 < split < 1.0, "Invalid Split {}".format(split)
            return self.images[:int(len(self.images)*split)], self.images[int(len(self.images)*split):]

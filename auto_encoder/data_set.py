import os
import cv2


class TrainImage:
    def __init__(self, base_path, image_name):
        self.path = os.path.join(base_path, image_name)
        self.base_path = base_path
        self.name = image_name
        self.h = None
        self.w = None

    def get_dims(self):
        if self.h is None or self.w is None:
            img = self.load_x()
            height, width, ch = img.shape
            self.h = height
            self.w = width
        return self.h, self.w

    def is_valid(self):
        img = self.load_x()
        if img is None:
            return False
        height, width, ch = img.shape
        if height < 10 or width < 10:
            return False
        return True

    def load_x(self):
        x = cv2.imread(self.path)
        assert x is not None, "No image was found."
        return x

    def get_label_file(self):
        df = os.path.dirname(self.base_path)
        return os.path.join(df, "labels", self.name[:-4] + ".txt")

    def load_y(self):
        pot_label_file = self.get_label_file()
        if os.path.isfile(pot_label_file):
            with open(pot_label_file) as f:
                content = f.read()
                label = content.strip()
                return label
        cls_name = os.path.basename(self.base_path)
        if cls_name == "images":
            previous_folder = os.path.dirname(self.base_path)
            cls_name = os.path.basename(previous_folder)
        return cls_name


def load_folder(path_to_folder):
    images = []
    for f in os.listdir(path_to_folder):
        if f.endswith((".png", ".jpg", ".jpeg", ".ppm")):
            train_img = TrainImage(path_to_folder, f)
            if train_img.is_valid():
                images.append(train_img)

        if os.path.isdir(os.path.join(path_to_folder, f)):
            sub_images = load_folder(os.path.join(path_to_folder, f))
            images += sub_images

    return images


class DataSet:
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.images = []

    def load(self):
        print("[INFO] Loading images from {} ...".format(self.path))
        self.images = load_folder(self.path)
        print("[INFO] Found: {} Images".format(len(self.images)))

    def get_data(self, split=0.0):
        if split == 0:
            return self.images
        else:
            assert 0 < split < 1.0, "Invalid Split {}".format(split)
            return self.images[:int(len(self.images)*split)], self.images[int(len(self.images)*split):]

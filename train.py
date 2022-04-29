
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.augmentations import Augmentations


class Config:
    def __init__(self):
        self.opt = {
            "backbone": "basic",
            "loss": "mse",
            "optimizer": "lazy_adam",
            "batch_size": 16,
            "embedding_size": 512,
            "init_learning_rate": 1e-5,
            "input_shape": [256, 256, 3],
        }


def main():
    ds = DataSet("/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/traffic_sign_classification/train/")
    ds.load()
    train_images, test_image = ds.get_data(0.8)

    ae = AutoEncoder("/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/ai_models/test_ae_2", Config())
    ae.build(True)
    augmentations = Augmentations()
    ae.fit(train_images, test_image, augmentations)


if __name__ == "__main__":
    main()

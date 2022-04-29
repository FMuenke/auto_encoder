import cv2
import os
import numpy as np
from auto_encoder.data_set import DataSet
from auto_encoder.auto_encoder import AutoEncoder

from auto_encoder.util import check_n_make_dir


class Config:
    def __init__(self):
        self.opt = {
            "backbone": "basic",
            "loss": "mse",
            "optimizer": "lazy_adam",
            "epochs": 10000,
            "batch_size": 16,
            "init_learning_rate": 1e-5,
            "input_shape": [256, 256, 3],
        }


def make_result_picture(img, res):
    res = res * 255
    img = cv2.resize(img, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)
    block = np.ones((img.shape[0], 10, 3))
    complete = np.concatenate([img, block, res], axis=1)
    return complete


def main():
    ds = DataSet("/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/traffic_sign_classification/test/0/images")
    ds.load()
    test_images = ds.get_data()

    model_path = "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/ai_models/test_ae"
    results_folder = os.path.join(model_path, "results")
    check_n_make_dir(results_folder, True)

    ae = AutoEncoder(model_path, Config())
    ae.build(False)

    for i in test_images:
        print(i.name)
        data = i.load_x()
        pred = ae.inference(data)
        pred = pred[0, :, :, :]

        cv2.imwrite(os.path.join(results_folder, i.name[:-4] + ".png"), make_result_picture(data, pred))


if __name__ == "__main__":
    main()

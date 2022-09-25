
from variational_auto_encoder.cnn_resnet import residual_auto_encoder


class Backbone:
    def __init__(self, backbone_type, embedding_size, depth, resolution, loss_type="mse", weights="imagenet"):
        self.backbone_type = backbone_type
        self.embedding_size = embedding_size

        self.depth = depth
        self.resolution = resolution

        self.loss_type = loss_type
        self.weights = weights

    def loss(self):
        if self.loss_type in ["mean_squared_error", "mse"]:
            return "mean_squared_error"

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def metric(self):
        return ["mse"]

    def build(self, input_shape, add_decoder=True):
        if self.backbone_type in ["resnet", "residual"]:
            encoder, decoder = residual_auto_encoder(
                input_shape=input_shape,
                embedding_size=self.embedding_size,
                depth=self.depth,
                resolution=self.resolution
            )
        else:
            raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

        if add_decoder:
            return decoder
        else:
            return encoder


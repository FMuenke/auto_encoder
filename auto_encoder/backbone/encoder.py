from tensorflow import keras
from auto_encoder.backbone.residual import make_residual_encoder


def get_encoder(
        backbone,
        input_shape,
        embedding_size,
        embedding_type,
        embedding_activation,
        depth,
        scale,
        resolution,
        drop_rate,
        dropout_structure
    ):
    if backbone in ["resnet", "residual"]:
        x_in, output = make_residual_encoder(
            input_shape=input_shape,
            embedding_size=embedding_size,
            embedding_type=embedding_type,
            embedding_activation=embedding_activation,
            depth=depth,
            scale=scale,
            resolution=resolution,
            drop_rate=drop_rate,
            dropout_structure=dropout_structure,
        )
    elif backbone in ["xception"]:
        base_model = keras.applications.xception.Xception(
            weights=None,
            include_top=False,
            pooling="avg",
            input_shape=(input_shape[0], input_shape[1], 3),
        )
        x_in, output = base_model.input, base_model.output
    elif backbone in ["resnet50"]:
        base_model = keras.applications.resnet.ResNet50(
            weights=None,
            include_top=False,
            pooling="avg",
            input_shape=(input_shape[0], input_shape[1], 3),
        )
        x_in, output = base_model.input, base_model.output
    elif backbone in ["efficientnetB0"]:
        base_model = keras.applications.efficientnet.EfficientNetB0(
            weights=None,
            include_top=False,
            pooling="avg",
            input_shape=(input_shape[0], input_shape[1], 3),
        )
        x_in, output = base_model.input, base_model.output
    else:
        raise ValueError("{} Backbone was not recognised".format(backbone))
    return x_in, output

from auto_encoder.auto_encoder import AutoEncoder
from variational_auto_encoder.backbone import Backbone


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, model_folder, opt):
        super(VariationalAutoEncoder, self).__init__(model_folder, opt)

    def build(self, compile_model=True, add_decoder=True):
        backbone = Backbone(
            self.backbone,
            embedding_size=self.embedding_size,
            depth=self.depth,
            resolution=self.resolution,
            loss_type=self.loss_type
        )
        x_input, y = backbone.build(self.input_shape, add_decoder=add_decoder)

        self.model = Model(inputs=x_input, outputs=y)
        # print(self.model.summary())

        self.load()
        if compile_model:
            self.model.compile(loss=backbone.loss(), optimizer=self.optimizer, metrics=backbone.metric())
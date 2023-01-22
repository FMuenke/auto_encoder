import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


class SimSiamEngine(tf.keras.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        projection_size = int(encoder.output.shape[-1])
        self.predictor = keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((projection_size,)),
                layers.Dense(
                    int(projection_size / 4),
                    use_bias=False,
                    kernel_regularizer=keras.regularizers.l2(0.0005),
                ),
                layers.ReLU(),
                layers.BatchNormalization(),
                layers.Dense(projection_size),
            ],
            name="predictor",
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (self.encoder.trainable_variables + self.predictor.trainable_variables)
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        p = self.predictor(z)
        return p

    def test_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        # Note that here we are enforcing the network to match
        # the representations of two differently augmented batches
        # of data.
        loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

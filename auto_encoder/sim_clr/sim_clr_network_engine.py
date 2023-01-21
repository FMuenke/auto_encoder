import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


# Distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, jitter, input_shape):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(input_shape[0], input_shape[1], 3)),
            RandomColorAffine(brightness, jitter),
        ]
    )


# Define the contrastive model with model-subclassing
class ContrastiveModel(keras.Model):
    def __init__(self, encoder, temperature, input_shape):
        super().__init__()
        projection_size = int(encoder.output.shape[-1])
        self.temperature = temperature
        contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2, "input_shape": input_shape}
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.encoder = encoder
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(projection_size,)),
                layers.Dense(projection_size, activation="relu"),
                layers.Dense(projection_size),
            ],
            name="projection_head",
        )
        # self.encoder.summary()
        # self.projection_head.summary()

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature)

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        images_1, images_2 = data

        # Both labeled and unlabeled images are used, without labels
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images_1, training=True)
        augmented_images_2 = self.contrastive_augmenter(images_2, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, mask=None):
        inputs = self.contrastive_augmenter(inputs)
        z = self.encoder(inputs)
        z = self.projection_head(z)
        return z

    # def test_step(self, data):
        # labeled_images, labels = data

        # For testing the components are used with a training=False flag
        # preprocessed_images = self.classification_augmenter(labeled_images, training=False)
        # features = self.encoder(preprocessed_images, training=False)
        #     class_logits = self.linear_probe(features, training=False)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # self.probe_loss_tracker.update_state(probe_loss)
        # self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        # return {m.name: m.result() for m in self.metrics[2:]}

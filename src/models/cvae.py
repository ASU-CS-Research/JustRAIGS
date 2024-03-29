from typing import Optional, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers
from wandb import Config
from wandb.sdk.wandb_run import Run


@tf.keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.saving.register_keras_serializable()
class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

@tf.keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

@tf.keras.saving.register_keras_serializable()
class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, batch_size: int,
        input_shape: Tuple[int, int, int], num_classes: int, original_dim: int, intermediate_dim: Optional[int] = 64,
        latent_dim: Optional[int] = 32, name: Optional[str] = "autoencoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._wab_trial_run = wab_trial_run
        self._trial_hyperparameters = trial_hyperparameters
        self._batch_size = batch_size
        self._input_shape_no_batch_dim = input_shape
        self._num_classes = num_classes
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

    def fit(self, x, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        logger.debug(f"Training model in .fit()...")
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        for epoch in range(epochs):
            logger.debug(f"Start of epoch {epoch}")
            for step, x_batch_train in enumerate(x):
                x_batch_train_images = x_batch_train[0]
                x_batch_train_labels = x_batch_train[1]
                with tf.GradientTape() as tape:
                    reconstructed = self(x_batch_train_images)
                    # Compute reconstruction loss
                    loss = tf.keras.losses.MeanSquaredError(x_batch_train_images, reconstructed)
                    loss += sum(self.losses)

                grads = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

                train_loss(loss)

                logger.debug(f"Step {step}, Loss: {train_loss.result()}")

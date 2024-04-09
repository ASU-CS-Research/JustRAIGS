from contextlib import redirect_stdout
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3
from loguru import logger
from wandb import Config
from wandb.sdk.wandb_run import Run
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import wandb as wab
import matplotlib.pyplot as plt
import time


@tf.keras.utils.register_keras_serializable(name='WaBModel')
class WaBModel(Sequential):
    """
    This is an example WaB model that is instantiated repeatedly by the :class:`~src.hypermodels.hypermodels.WaBHyperModel`.
    This class encapsulates model construction and save and restore logic, which is particularly useful when leveraging
    custom layers or metrics.
    """
    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, batch_size: int,
            input_shape: Tuple[int, int, int], *args, **kwargs):
        """

        Args:
            wab_trial_run (Optional[Run]): The WaB run object that is responsible for logging the results of the current
              trial. Used to log output to the same namespaced location in WaB. Note that this parameter is optional in the
              event that the model is being loaded from a saved model format (e.g. h5) in which case the user may not wish
              to log metrics to the same trial as the one that generated the saved model. During training it is expected
              that this value is not ``None``.
            trial_hyperparameters (Config): The hyperparameters for this particular trial. These are provided by the WaB
              agent that is driving the sweep as a subset of the total hyperparameter search space.
            batch_size (int): The batch size to use for the training model.
            input_shape (Tuple[int, int, int]): The shape of the input tensor WITHOUT the batch dimension (that means no
              leading batch dimension integer and no leading ``None`` placeholder Tensor).
            *args: Variable length argument list to pass through to the :class:`~tf.keras.Model` superclass constructor.
            **kwargs: Arbitrary keyword arguments to pass through to the :class:`~tf.keras.Model` superclass constructor.

        Notes:
            If you are wondering about the usage of the decorator on this class see: https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects

        """
        self._wab_trial_run = wab_trial_run
        logger.debug(f"Initialize via call to super()...")
        super().__init__(*args, **kwargs)
        self._trial_hyperparameters = trial_hyperparameters
        self._batch_size = batch_size
        self._input_shape_no_batch = input_shape
        # Build the model with the hyperparameters for this particular trial.
        # Note: The concrete subclass should know which hyperparameters are pertinent to it for the particular trial.
        kernel_size = trial_hyperparameters['kernel_size']
        num_nodes_conv2d_1 = trial_hyperparameters['num_nodes_conv_1']
        if 'num_nodes_conv_2' in trial_hyperparameters:
            num_nodes_conv2d_2 = trial_hyperparameters['num_nodes_conv_2']
        else:
            num_nodes_conv2d_2 = None
        conv_layer_activation_function = trial_hyperparameters['conv_layer_activation_function']
        self._input_layer = tf.keras.layers.InputLayer(
            input_shape=self._input_shape_no_batch, batch_size=self._batch_size, name='input_layer'
        )
        self._conv_2d_1 = tf.keras.layers.Conv2D(
            filters=num_nodes_conv2d_1, kernel_size=kernel_size, activation=conv_layer_activation_function,
            padding='same', name='conv_2d_1'
        )
        if num_nodes_conv2d_2 is not None:
            self._conv_2d_2 = tf.keras.layers.Conv2D(
                filters=num_nodes_conv2d_2, kernel_size=kernel_size, activation=conv_layer_activation_function,
                padding='same', name='conv_2d_2'
            )
        self._average_pool_2d_1 = tf.keras.layers.AveragePooling2D(
            pool_size=(1, self._input_shape_no_batch[0]),
            strides=(1, self._input_shape_no_batch[0]),
            padding='same',
            name='average_pool_2d_1'
        )
        self._flatten_1 = tf.keras.layers.Flatten(name='flatten_1')
        self._output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        logger.debug(f"Constructing model...")
        self.add(self._input_layer)
        self.add(self._conv_2d_1)
        if num_nodes_conv2d_2 is not None:
            self.add(self._conv_2d_2)
        self.add(self._average_pool_2d_1)
        self.add(self._flatten_1)
        self.add(self._output_layer)
        logger.debug(f"Setting up output directories...")
        self._repo_root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../"))

    def get_config(self):
        """
        Utilized to return a serialized representation of the model. This is used when restoring the model from disk.

        See Also:
            - https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects

        """
        base_config = super().get_config()
        # .. todo:: Should the wab_trial_run object be serialized in the config?
        config = {
            'wab_trial_run': None, 'trial_hyperparameters': self._trial_hyperparameters.as_dict(),
            'input_shape': self._input_shape_no_batch, 'batch_size': self._batch_size
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """

        Args:
            config (Dict[str, Any]): The configuration dictionary used to construct the model from h5 format.

        See Also:
            https://www.tensorflow.org/guide/keras/serialization_and_saving#custom_objects

        Returns:
            WaBModel: The model constructed from the provided configuration dictionary. Weights will be
            restored.

        """
        logger.debug(f"Deserializing from config: {config}")
        trial_hyperparameters_dict = config.pop('trial_hyperparameters')
        input_shape = config.pop('input_shape')
        batch_size = config.pop('batch_size')
        layers = config.pop('layers')
        wab_trial_run = config.pop('wab_trial_run')
        trial_hyperparameters = Config()
        trial_hyperparameters.update(trial_hyperparameters_dict)
        return cls(
            wab_trial_run=None, trial_hyperparameters=trial_hyperparameters, input_shape=input_shape,
            batch_size=batch_size, **config
        )

    def save(self, *args, **kwargs):
        """
        Saves the model to disk preserving the trained weights. Loads the saved model back into memory immediately to
        ensure that the weights were saved and deserialized correctly.

        See Also:
            - https://www.tensorflow.org/guide/keras/serialization_and_saving

        """
        logger.debug(f"save method received args: {args}")
        logger.debug(f"save method received kwargs: {kwargs}")
        saved_model_path = args[0]
        # saved_model_path = saved_model_path.replace('.h5', '.keras') overwrite = kwargs['overwrite']
        if 'save_format' in kwargs:
            save_format = kwargs['save_format']
        else:
            # When save_model is called with no save_format kwarg for the .h5 format:
            save_format = 'h5'
        if os.path.isfile(saved_model_path):
            super().save(saved_model_path, **kwargs)
            logger.debug(f"Overwrote and saved model to: {saved_model_path}")
        else:
            super().save(saved_model_path, **kwargs)
            logger.debug(f"Saved model to: {saved_model_path}")
        if save_format == 'h5':
            # Load in saved model and run assertions:
            logger.debug(f"Loading saved model for weight assertion check...")
            loaded_model = tf.keras.models.load_model(
                args[0], custom_objects={"WaBModel": WaBModel}
            )
            error_message = f"Saved model weight assertion failed. Weights were most likely saved incorrectly"
            np.testing.assert_equal(self.get_weights(), loaded_model.get_weights()), error_message
            saved_model_artifact = wab.Artifact("saved_model.h5", "saved_model")
            saved_model_artifact.add_file(saved_model_path)
            self._wab_trial_run.log_artifact(saved_model_artifact)
        elif save_format == 'tf':
            logger.warning(f"TensorFlow model format (.tf) save-and-restore logic is not yet working. Anticipate an "
                           f"un-deserializable model.")
        else:
            logger.error(f"Unsupported save_format: {save_format}. Model was not saved.")


@tf.keras.utils.register_keras_serializable(name='InceptionV3WaBModel')
class InceptionV3WaBModel(Model):
    """
    A WaBModel that is constructed from a pretrained InceptionV3 model. This class is a useful reference for how to
    implement transfer learning with WaB.
    """

    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, batch_size: int,
            input_shape: Tuple[int, int, int], num_classes: int, *args, **kwargs):
        """

        Args:
            wab_trial_run (Optional[Run]): The WaB run object that is responsible for logging the results of the current
              trial. Used to log output to the same namespaced location in WaB. Note that this parameter is optional in the
              event that the model is being loaded from a saved model format (e.g. h5) in which case the user may not wish
              to log metrics to the same trial as the one that generated the saved model. During training it is expected
              that this value is not ``None``.
            trial_hyperparameters (Config): The hyperparameters for this particular trial. These are provided by the WaB
              agent that is driving the sweep as a subset of the total hyperparameter search space.
            batch_size (int): The batch size to use for the training model.
            input_shape (Tuple[int, int, int]): The shape of the input tensor WITHOUT the batch dimension (that means no
              leading batch dimension integer and no leading ``None`` placeholder Tensor).
            num_classes (int): The number of classes in the classification task, used to construct the output layer of
              the model and decide whether to use sigmoid or softmax.
            *args: Variable length argument list to pass through to the :class:`keras.Model` superclass constructor.
            **kwargs: Arbitrary keyword arguments to pass through to the :class:`keras.Model` superclass constructor.

        Notes:
            If you are wondering about the usage of the decorator on this class see: https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects

        """
        self._wab_trial_run = wab_trial_run
        logger.debug(f"Initialize via call to super()...")
        super().__init__(*args, **kwargs)
        self._trial_hyperparameters = trial_hyperparameters
        self._batch_size = batch_size
        self._input_shape_no_batch = input_shape
        self._num_classes = num_classes
        self._base_model = InceptionV3(
            include_top=False, weights='imagenet', input_shape=self._input_shape_no_batch, classes=self._num_classes
        )
        # Freeze the base model:
        for layer in self._base_model.layers:
            layer.trainable = False
        '''
        Build the model with the hyperparameters for this particular trial:
        '''
        # Ensure the necessary hyperparameters are present in the search space:
        if 'num_thawed_layers' in self._trial_hyperparameters:
            for i in range(self._trial_hyperparameters['num_thawed_layers']):
                self._base_model.layers[-i].trainable = True
        else:
            raise ValueError(f"num_thawed_layers not found in trial hyperparameters: {self._trial_hyperparameters}")
        if 'optimizer' not in self._trial_hyperparameters:
            raise ValueError(f"Optimizer not found in trial hyperparameters: {self._trial_hyperparameters}")
        if 'loss' not in self._trial_hyperparameters:
            raise ValueError(f"Loss function not found in trial hyperparameters: {self._trial_hyperparameters}")
        # Parse Optimizer:
        optimizer_config = self._trial_hyperparameters['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            self._optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            self._optimizer = None
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # Parse loss function:
        loss_function = self._trial_hyperparameters['loss']
        if loss_function == 'binary_crossentropy':
            self._loss = BinaryCrossentropy(from_logits=False)
        else:
            logger.error(f"Unknown loss function: {loss_function} provided in the hyperparameter section of the sweep "
                         f"configuration.")
            exit(1)
        # Add data augmentation layers
        input_layer = tf.keras.Input(self._input_shape_no_batch)
        random_flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")(input_layer)
        random_rotate = tf.keras.layers.RandomRotation(0.2)(random_flip)
        self._base_model = self._base_model(random_rotate)
        # Add a new head to the model (i.e. new Dense fully connected layyer and softmax):
        model_head = Flatten()(self._base_model)
        model_head = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid')(model_head)
        self._model = Model(inputs=input_layer, outputs=model_head)
        # Build the model:
        self._model.build((None,) + self._input_shape_no_batch)
        # Log the model summary to WaB:
        self._wab_trial_run.log({"model_summary": self._model.summary()})
        # Compile the model:
        self._model.compile(loss=self._loss, optimizer=self._optimizer)
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def get_config(self):
        """
        Utilized to return a serialized representation of the model. This is used when restoring the model from disk.

        .. todo:: Remove duplicate logic by refactoring the WaBModel as a superclass.

        See Also:
            - https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects

        """
        base_config = super().get_config()
        # .. todo:: Should the wab_trial_run object be serialized in the config?
        config = {
            'wab_trial_run': None, 'trial_hyperparameters': self._trial_hyperparameters.as_dict(),
            'input_shape': self._input_shape_no_batch, 'batch_size': self._batch_size
        }
        return {**base_config, **config}

    def save(self, *args, **kwargs):
        """
        Saves the model to disk preserving the trained weights. Loads the saved model back into memory immediately to
        ensure that the weights were saved and deserialized correctly.

        .. todo:: Remove duplicate code by refactoring the WaBModel as a superclass.

        See Also:
            - https://www.tensorflow.org/guide/keras/serialization_and_saving

        """
        logger.debug(f"save method received args: {args}")
        logger.debug(f"save method received kwargs: {kwargs}")
        saved_model_path = args[0]
        # saved_model_path = saved_model_path.replace('.h5', '.keras')
        # overwrite = kwargs['overwrite']
        if 'save_format' in kwargs:
            save_format = kwargs['save_format']
        else:
            # When save_model is called with no save_format kwarg for the .h5 format:
            save_format = 'h5'
        if os.path.isfile(saved_model_path):
            super().save(saved_model_path, **kwargs)
            logger.debug(f"Overwrote and saved model to: {saved_model_path}")
        else:
            super().save(saved_model_path, **kwargs)
            logger.debug(f"Saved model to: {saved_model_path}")
        if save_format == 'h5':
            # Load in saved model and run assertions:
            logger.debug(f"Loading saved model for weight assertion check...")
            loaded_model = tf.keras.models.load_model(
                args[0], custom_objects={"InceptionV3WaBModel": WaBModel}
            )
            # loaded_model.compile(optimizer=self._trial_hyperparameters['optimizer'], loss='binary_crossentropy')
            error_message = f"Saved model weight assertion failed. Weights were most likely saved incorrectly"
            np.testing.assert_equal(self.get_weights(), loaded_model.get_weights()), error_message
            saved_model_artifact = wab.Artifact("saved_model.h5", "saved_model")
            saved_model_artifact.add_file(saved_model_path)
            self._wab_trial_run.log_artifact(saved_model_artifact)
        elif save_format == 'tf':
            logger.warning(f"TensorFlow model format (.tf) save-and-restore logic is not yet working. Anticipate an "
                           f"un-deserializable model.")
        else:
            logger.error(f"Unsupported save_format: {save_format}. Model was not saved.")




class CVAEWaBModel(Model):
    """
    A Convolutional Variational Autoencoder (CVAE) model that is intended to be used as a feature extractor on the raw
    high-dimensional input images. This model is part of the demo pipeline for performing feature extraction prior to
    classification within the WaB framework.

    See Also:
        - https://www.tensorflow.org/tutorials/generative/cvae
        - https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#putting_it_all_together_an_end-to-end_example

    """

    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, batch_size: int,
            input_shape: Tuple[int, int, int], num_classes: int, *args, **kwargs):
        """

        .. todo:: Remove duplicate code by subclassing from a common WaBModel base class.

        .. todo:: Add additional utility methods to this class from: https://www.tensorflow.org/tutorials/generative/cvae

        Args:
            wab_trial_run (Optional[Run]): The WaB run object that is responsible for logging the results of the current
              trial. Used to log output to the same namespaced location in WaB. Note that this parameter is optional in
              the event that the model is being loaded from a saved model format (e.g. h5) in which case the user may
              not wish to log metrics to the same trial as the one that generated the saved model. During training it is
              expected that this value is not ``None``.
            trial_hyperparameters (Config): The hyperparameters for this particular trial. These are provided by the WaB
              agent that is driving the sweep as a subset of the total hyperparameter search space.
            batch_size (int): The batch size to use for the training model.
            input_shape (Tuple[int, int, int]): The shape of the input tensor WITHOUT the batch dimension (that means no
              leading batch dimension integer and no leading ``None`` placeholder Tensor).
            num_classes (int): The number of classes in the classification task, used to construct the output layer of
              the model and decide whether to use sigmoid or softmax.
            *args: Variable length argument list to pass through to the :class:`keras.Model` superclass constructor.
            **kwargs: Arbitrary keyword arguments to pass through to the :class:`keras.Model` superclass constructor.

        """
        self._wab_trial_run = wab_trial_run
        super().__init__(*args, **kwargs)
        self._trial_hyperparameters = trial_hyperparameters
        self._batch_size = batch_size
        self._input_shape_no_batch = input_shape
        self._num_classes = num_classes
        '''
        Build the model with the hyperparameters for this particular trial:
        '''
        # Ensure the necessary hyperparameters are present in the search space:
        if 'optimizer' not in self._trial_hyperparameters:
            raise ValueError(f"Optimizer not found in trial hyperparameters: {self._trial_hyperparameters}")
        if 'feature_extraction' not in self._trial_hyperparameters:
            raise ValueError(f"Feature extraction not found in trial hyperparameters: {self._trial_hyperparameters}")
        # Parse Optimizer:
        optimizer_config = self._trial_hyperparameters['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            self._optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            self._optimizer = None
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # Parse Feature Extraction hyperparameters:
        feature_extraction_config = self._trial_hyperparameters['feature_extraction']
        if 'latent_dim' in feature_extraction_config:
            self._latent_dim = feature_extraction_config['latent_dim']
        else:
            raise ValueError(f"latent_dim not found in trial hyperparameters: {self._trial_hyperparameters}")
        if 'loss' in feature_extraction_config:
            loss = feature_extraction_config['loss']
            if loss == 'mean':
                self._loss = tf.keras.metrics.Mean()
            else:
                raise ValueError(f"Unknown loss function: {loss} provided in the hyperparameter section of the sweep "
                                 f"configuration.")
        else:
            raise ValueError(f"Loss function not found in trial hyperparameters for feature extraction: "
                             f"{self._trial_hyperparameters}")
        self._encoder = Sequential([
            # ..todo:: Dynamic input shape:
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(self._latent_dim + self._latent_dim)
        ])
        self._decoder = Sequential([
            tf.keras.layers.InputLayer(input_shape=(self._latent_dim,)),
            tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', activation='relu'
            ),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'
            )
        ])

    @tf.function
    def sample(self, eps: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Sample the latent embedding space to generate a sample for the decoder.

        Args:
            eps (Optional[float]): Epsilon constant in the equation `z = mu + sigma * epsilon`. If None, then a random
              epsilon is sampled from a standard normal distribution the size of the latent representation.

        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self._latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Encode the input tensor into a latent representation.

        Args:
            x (:class:`~tensorflow.Tensor`): The input tensor to encode into a latent representation.

        """
        mean, logvar = tf.split(self._encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        """
        Re-parameterize the latent representation to sample from the distribution specified by the provided ``mean`` and
        log of the variance ``logvar``.

        Args:
            mean (:class:`~tensorflow.Tensor`): The mean of the distribution.
            logvar (:class:`~tensorflow.Tensor`): The log variance of the distribution.

        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
        """
        Decode the latent representation into a reconstructed input tensor.

        Args:
            z (:class:`~tensorflow.Tensor`): The latent representation to decode.
            apply_sigmoid (bool): Whether to apply the sigmoid function to the output tensor.

        """
        logits = self._decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def log_normal_pdf(self, sample: tf.Tensor, mean: Union[tf.Tensor, float], logvar: Union[tf.Tensor, float], raxis: Optional[int] = 1) -> tf.Tensor:
        """
        Compute the log probability density function of the normal distribution.

        Args:
            sample (:class:`~tensorflow.Tensor`): The sample to compute the log probability density function for.
            mean (:class:`~tensorflow.Tensor`): The mean of the normal distribution.
            logvar (:class:`~tensorflow.Tensor`): The log variance of the normal distribution.
            raxis (Optional[int]): The axis to reduce over, defaults to axis ``1`` if not provided.

        """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_loss(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the loss for the CVAE model. This is the Monte Carlo estimate of the negative evidence lower bound
        (ELBO).

        .. todo:: Docstring.

        """
        logger.debug(f"compute_loss for x.shape: {x.shape}")
        mean, logvar = self.encode(x)
        logger.debug(f"mean.shape, logvar.shape: {mean.shape}, {logvar.shape}")
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        return loss

    @tf.function
    def train_step(self, x: tf.Tensor) -> tf.Tensor:
        """
        A single training step for the CVAE model.

        Args:
            x (:class:`~tensorflow.Tensor`): The input tensor to train the model on.

        Returns:
           (:class:`~tensorflow.Tensor`): The loss for the current training step.

        """
        if type(x) is tuple:
            image = x[0]
            label = x[1]
            # Prepend batch dimension:
            # image = tf.reshape(image, (1,) + image.shape)
        else:
            image = x
        # logger.debug(f"Training step image.shape: {image.shape}")
        with tf.GradientTape() as tape:
            train_loss = self.compute_loss(image)
        gradients = tape.gradient(train_loss, self.trainable_variables)
        # Gradient clipping to 5.0 for extremely large negative gradients:
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return train_loss

    def fit(
            self, x, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        """
        See Also:
            - https://www.tensorflow.org/tutorials/generative/cvae

        """
        logger.debug(f"Training model in .fit ...")
        for i in range(1, epochs + 1):
            start_time = time.time()
            for train_image, train_label in x:
                train_loss = self.train_step(train_image)
                self._wab_trial_run.log({"loss": train_loss})
                self._loss(train_loss)
            train_elbo = -self._loss.result()
            self._wab_trial_run.log({"ELBO": train_elbo})
            end_time = time.time()
            # Reset state for validation loss:
            self._loss.reset_state()
            for val_image, val_label in validation_data:
                val_loss = self.compute_loss(val_image)
                self._wab_trial_run.log({"val_loss": val_loss})
                self._loss(val_loss)
            val_elbo = -self._loss.result()
            self._wab_trial_run.log({"val_ELBO": val_elbo})
            print(f"Epoch: {i}, Train set ELBO: {train_elbo}, Validation set ELBO: {val_elbo}, time elapse for current epoch: {end_time - start_time}")

    def call(self, inputs, training=None, mask=None):
        """

        """
        logger.debug(f"inputs: {inputs}")
        self.train_step(inputs)
        # self._loss(self.compute_loss(inputs))
        # elbo = -self._loss.result()
        return

    def generate_and_save_images(self, epoch: int, test_sample: tf.Tensor):
        mean, logvar = self.encode(test_sample)
        z = self.reparameterize(mean, logvar)
        predictions = self.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig(f'image_at_epoch_{epoch:04d}.png')
        plt.show()

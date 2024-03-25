from contextlib import redirect_stdout
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB7
from loguru import logger
from wandb import Config
from wandb.sdk.wandb_run import Run
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import BinaryCrossentropy
import wandb as wab


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


@tf.keras.utils.register_keras_serializable(name='EfficientNetB7WaBModel')
class EfficientNetB7WaBModel(Model):
    """
    A WaBModel that is constructed from a pretrained EfficientNetB7 model. This class is a useful reference for how to
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
        self._base_model = EfficientNetB7(
            include_top=False, weights='imagenet', input_shape=self._input_shape_no_batch, classes=self._num_classes
        )
        # Freeze the base model:
        for layer in self._base_model.layers:
            layer.trainable = True
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
        elif optimizer_type == 'sgd':
            self._optimizer = SGD(learning_rate=optimizer_learning_rate)
        elif optimizer_type == 'rmsprop':
            self._optimizer = RMSprop(learning_rate=optimizer_learning_rate)
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
        # Add a new head to the model (i.e. new Dense fully connected layer and softmax):
        model_head = Flatten()(self._base_model.outputs[0])
        model_head = tf.keras.layers.Dense(512, activation='relu')(model_head)
        model_head = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid')(model_head)
        self._model = Model(inputs=self._base_model.inputs, outputs=model_head)
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
                args[0], custom_objects={"EfficientNetB7WaBModel": WaBModel}
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

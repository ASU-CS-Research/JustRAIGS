from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from loguru import logger
from wandb import Config
from wandb.sdk.wandb_run import Run
import os
from tensorflow.keras.models import Sequential


@tf.keras.utils.register_keras_serializable(name='WaBModel')
class WaBModel(Sequential):
    """
    See sweeper_old.py PipingDetectorWabModel

    This is an example WaB model that is instantiated repeatedly by the :class:`hypermodels.hypermodel.WaBHyperModel`.
    This class encapsulates model construction and save and restore logic, which is particularly useful when leveraging
    custom layers or metrics.
    """
    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, batch_size: int,
            input_shape: Tuple[int, int, int], *args, **kwargs):
        """
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
        *args: Variable length argument list to pass through to the :class:`keras.Model` superclass constructor.
        **kwargs: Arbitrary keyword arguments to pass through to the :class:`keras.Model` superclass constructor.

        Notes::
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
            'input_shape': self._input_shape_no_batch
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
        layers = config.pop('layers')
        wab_trial_run = config.pop('wab_trial_run')
        trial_hyperparameters = Config()
        trial_hyperparameters.update(trial_hyperparameters_dict)
        return cls(wab_trial_run=None, trial_hyperparameters=trial_hyperparameters, input_shape=input_shape, **config)

    def save(self, *args, **kwargs):
        """
        Saves the model to disk preserving the trained weights.

        See Also:
            - https://www.tensorflow.org/guide/keras/serialization_and_saving

        """
        logger.debug(f"save method received args: {args}")
        logger.debug(f"save method received kwargs: {kwargs}")
        saved_model_path = args[0]
        # saved_model_path = saved_model_path.replace('.h5', '.keras')
        overwrite = kwargs['overwrite']
        # if 'save_format' in kwargs:
        #     kwargs['save_format'] = 'keras'
        if os.path.isfile(saved_model_path):
            if overwrite:
                super().save(saved_model_path, **kwargs)
                logger.debug(f"Overwrote and saved model to: {saved_model_path}")
        else:
            super().save(saved_model_path, **kwargs)
            logger.debug(f"Saved model to: {saved_model_path}")
        # Load in saved model and run assertions:
        logger.debug(f"Loading saved model for weight assertion check...")
        loaded_model = tf.keras.models.load_model(
            args[0], custom_objects={"WaBModel": WaBModel}
        )
        # loaded_model.compile(optimizer=self._trial_hyperparameters['optimizer'], loss='binary_crossentropy')
        error_message = f"Saved model weight assertion failed. Weights were most likely saved incorrectly"
        np.testing.assert_equal(self.get_weights(), loaded_model.get_weights()), error_message

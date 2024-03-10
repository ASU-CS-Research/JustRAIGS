import argparse
import enum
import os
import shutil
import socket
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, List, Union
import numpy as np
import wandb.util
from keras.layers import Flatten
from keras.metrics import Metric
from loguru import logger
import keras_tuner as kt
import tensorflow as tf
import copy
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
from tensorflow.keras import Model
from keras.optimizers import Optimizer
from keras.engine import base_layer, data_adapter, training_utils
from keras.utils import version_utils, tf_utils, io_utils
from keras import callbacks as callbacks_module, Input
from tensorflow.python.eager import context
from keras import optimizers
from wandb.sdk.wandb_run import Run
from keras.losses import Loss, MAE, MAPE, MSE
from tensorflow.keras.applications.vgg19 import VGG19
import wandb as wab
from src.piping_detection.metrics import BalancedBinaryAccuracy
from src.tuners.wab_kt_tuner import WBTuner
from src.MongoUtils.beedb import BeeDBUtils
from src.MongoUtils.stft_params import StftParams, AudioLib

# from metrics import BalancedBinaryAccuracy
# from wb_tuner import WBTuner

REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../"))
DATA_DIR = os.path.abspath(os.path.join(REPO_ROOT_DIR, "data"))
DATA_TEMP_DIR = os.path.abspath(os.path.join(DATA_DIR, "temp"))

if not os.environ.get("TFHUB_CACHE_DIR"):
    os.environ.setdefault("TFHUB_CACHE_DIR", f"{DATA_TEMP_DIR}/tfhub")

if not os.environ.get("WANDB_CONFIG_DIR"):
    os.environ.setdefault("WANDB_CONFIG_DIR", f"{DATA_TEMP_DIR}/wandb")

if socket.gethostname() == 'appmais.appstate.edu' or socket.gethostname() == 'lambda':
    mongo_utils_dir = os.path.abspath(os.path.join(__file__, '../../MongoUtils'))
    assert os.path.exists(mongo_utils_dir), f"{mongo_utils_dir} does not exist."
    sys.path.append(mongo_utils_dir)

# Temporary hack for pdb not resolving module imports the same way as normal python:
# if socket.gethostname() == 'appmais.appstate.edu' or socket.gethostname() == 'lambda':
#     src_dir = os.path.abspath(os.path.join(__file__, '../../../src'))
#     assert os.path.exists(src_dir), f"{src_dir} does not exist."
#     sys.path.append(src_dir)

VGG19_INPUT_SHAPE = os.environ.get("VGG19_INPUT_SHAPE", (224, 224))
BATCH_SIZE = os.environ.get("BATCH_SIZE", 64)
SEED = os.environ.get("SEED", 42)
GRAYSCALE = os.environ.get("GRAYSCALE", True)


class DatasetSplit(enum.Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class PipingDetectorHyperModel(kt.HyperModel):
    """
    This code is designed to run on the lambda machine compute server with an SSHFS mount to the AppMAIS data storage
    server.
    """

    def __init__(self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
                 metrics: List[Union[Metric, str]],
                 hyperparameters: Optional[kt.HyperParameters] = None, resampled_steps_per_epoch: Optional[int] = None):
        """
        This method should accept hyperparameters which remain constant during hyperparameter tuning.

        .. todo:: Docstrings.

        Args:
             resampled_steps_per_epoch (Optional[int]): In the event that the data has been resampled (i.e. for class
               balancing) then this value is expected to be provided. It indicates how many steps per epoch should be
               taken before every element of the positive/majority class will have been seen at least once. This
               information is required because during the oversampling process the validation and training datasets are
               made to have an infinite cardinality.

        """
        self._input_image_shape = input_image_shape
        self._vgg_19_input_image_shape = (224, 224, 3)
        self._num_classes = num_classes
        self._optimizer = None
        self._loss = loss
        self._metrics = metrics
        self.hyperparameters = hyperparameters
        self._resampled_steps_per_epoch = resampled_steps_per_epoch
        self._model = None
        super().__init__(name='PipingDetectorHyperModel', tunable=True)

    # @override
    def build(self, hp: kt.HyperParameters, weights: Optional[str] = 'imagenet', *args, **kwargs) -> Model:
        """
        This method setups up the hyperparameter grid and search space.

        See Also:
            https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras
        """
        logger.info(f"Building hypermodel with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._vgg_19_input_image_shape)
        preprocessing_layer = tf.keras.applications.vgg19.preprocess_input(input_layer)
        base_model = VGG19(include_top=False, weights=weights, input_shape=self._vgg_19_input_image_shape,
                           classes=self._num_classes)
        base_model.layers[0] = preprocessing_layer
        # Freeze the base model:
        for layer in base_model.layers:
            layer.trainable = False
        # Thaw the last layers if specified in the hyperparameters:
        if 'num_thawed_layers' in hp:
            for i in range(hp['num_thawed_layers']):
                base_model.layers[-1 * i].trainable = True
        # Add new head to the model (i.e. new Dense fully connected layer and softmax):
        model_head = Flatten()(base_model.outputs[0])
        model_head = tf.keras.layers.Dense(self._num_classes - 1, activation='softmax')(model_head)
        model = Model(inputs=base_model.inputs, outputs=model_head)
        model.build((None,) + self._vgg_19_input_image_shape)
        # compile the model:
        model.summary()
        # Reset the optimizer state (from any previous trials):
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        # self._optimizer.build(model.trainable_variables)
        return model

    @property
    def optimizer(self) -> Optional[Optimizer]:
        return self._optimizer

    @staticmethod
    def _disallow_inside_tf_function(method_name):
        if tf.inside_function():
            error_msg = (
                "Detected a call to `Model.{method_name}` inside a `tf.function`. "
                "`Model.{method_name} is a high-level endpoint that manages its "
                "own `tf.function`. Please move the call to `Model.{method_name}` "
                "outside of all enclosing `tf.function`s. Note that you can call a "
                "`Model` directly on `Tensor`s inside a `tf.function` like: "
                "`model(x)`."
            ).format(method_name=method_name)
            raise RuntimeError(error_msg)

    @staticmethod
    def _get_verbosity(verbose, distribute_strategy):
        """Find the right verbosity value for 'auto'."""
        if verbose == 1 and distribute_strategy._should_use_with_coordinator:
            raise ValueError(
                "`verbose=1` is not allowed with `ParameterServerStrategy` for "
                f"performance reasons. Received: verbose={verbose}"
            )
        if verbose == "auto":
            if (
                    distribute_strategy._should_use_with_coordinator
                    or not io_utils.is_interactive_logging_enabled()
            ):
                # Defaults to epoch-level logging for PSStrategy or using absl
                # logging.
                return 2
            else:
                return 1  # Defaults to batch-level logging otherwise.
        return verbose

    # @override
    def fit(self, model: Model, weights_and_biases_trial_run_object: Run, x=None, y=None, batch_size=None, epochs=1,
            verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_batch_size=None, validation_freq=1, train_eval_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False):
        """
        Overrides the :meth:`fit` method to enable Weights and Biases callback logging per-batch and per-epoch.
        """
        base_layer.keras_api_gauge.get_cell("fit").set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph("Model", "fit")
        model._assert_compile_was_called()
        model._check_call_args("fit")
        self._disallow_inside_tf_function("fit")
        verbose = self._get_verbosity(verbose, model.distribute_strategy)
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for `Tensor` and `NumPy` input.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )
        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter.unpack_x_y_sample_weight(validation_data)
        if model.distribute_strategy._should_use_with_coordinator:
            model._cluster_coordinator = (
                tf.distribute.experimental.coordinator.ClusterCoordinator(
                    model.distribute_strategy
                )
            )
        with model.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(  # noqa: E501
                model
        ):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.get_data_handler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=model,
                steps_per_execution=model._steps_per_execution,
            )
            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=model,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps,
                )
            stop_training = False
            train_function = model.make_train_function()
            model._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            steps_per_epoch_inferred = (
                    steps_per_epoch or data_handler.inferred_steps
            )
            (
                data_handler._initial_epoch,
                data_handler._initial_step,
            ) = model._maybe_load_initial_counters_from_ckpt(
                steps_per_epoch_inferred, initial_epoch
            )
            logs = None
            for epoch, iterator in data_handler.enumerate_epochs():
                model.reset_metrics()
                callbacks.on_epoch_begin(epoch, logs)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with tf.profiler.experimental.Trace(
                                "train",
                                epoch_num=epoch,
                                step_num=step,
                                batch_size=batch_size,
                                _r=1,
                        ):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = model.train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            # No error, now safe to assign to logs.
                            logs = tmp_logs
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                            if train_eval_freq % (step + 1) == 0:
                                # Log the training batch loss metrics to WANDB:
                                weights_and_biases_trial_run_object.log({f"train_batch_log": logs})
                            if stop_training:
                                break
                logs = tf_utils.sync_to_numpy_or_python_type(logs)
                if logs is None:
                    raise ValueError(
                        "Unexpected result of `train_function` "
                        "(Empty logs). Please use "
                        "`Model.compile(..., run_eagerly=True)`, or "
                        "`tf.config.run_functions_eagerly(True)` for more "
                        "information of where went wrong, or file a "
                        "issue/bug to `tf.keras`."
                    )
                # Override with model metrics instead of last step logs
                logs = model._validate_and_get_metrics_result(logs)
                epoch_logs = copy.copy(logs)
                # Log the epoch loss metrics (on the training data) to WANDB:
                weights_and_biases_trial_run_object.log({f"train_epoch_log": epoch_logs})
                # Run validation.
                if validation_data and model._should_eval(
                        epoch, validation_freq
                ):
                    # Check to see if the validation dataset is infinite:
                    # size = tf.data.experimental.cardinality(val_x).numpy()
                    # if size == tf.data.experimental.INFINITE_CARDINALITY:
                    #     eval_steps = self._resampled_steps_per_epoch
                    # else:
                    #     eval_steps = None
                    # Create data_handler for evaluation and cache it.
                    if getattr(model, "_eval_data_handler", None) is None:
                        model._eval_data_handler = data_adapter.get_data_handler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=model,
                            steps_per_execution=model._steps_per_execution
                        )
                    val_logs = model.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {
                        "val_" + name: val for name, val in val_logs.items()
                    }
                    weights_and_biases_trial_run_object.log({f"val_epoch_log": val_logs})
                    epoch_logs.update(val_logs)
                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if stop_training:
                    break
            if isinstance(self._optimizer, Optimizer) and epochs > 0:
                # if epochs > 0:
                self._optimizer.finalize_variable_values(
                    model.trainable_variables
                )
            # If eval data_handler exists, delete it after all epochs are done.
            if getattr(model, "_eval_data_handler", None) is not None:
                del model._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return model.history

    @property
    def model(self) -> Optional[Model]:
        return self._model


class FromScratch2DConvHyperModel(PipingDetectorHyperModel):

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]], hyperparameters: Optional[kt.HyperParameters] = None,
            resampled_steps_per_epoch: Optional[int] = None):
        super().__init__(input_image_shape, num_classes, loss, metrics, hyperparameters, resampled_steps_per_epoch)

    def build(self, hp: kt.HyperParameters, weights: Optional[str] = 'imagenet', *args, **kwargs) -> Model:
        logger.info(f"Building hyperparameter with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._input_image_shape)
        if self._num_classes == 2:
            # The final dense layer should just be a single neuron with a sigmoid activation function.
            output_layer = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid', name='output_layer')
        else:
            output_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax', name='output_layer')
        kernel_size = hp['kernel_size']
        conv_layer_activation_function = hp['conv_layer_activation_function']
        num_nodes_first_conv_layer = hp['num_nodes_conv_1']
        try:
            num_nodes_second_conv_layer = hp['num_nodes_conv_2']
        except KeyError:
            num_nodes_second_conv_layer = None
        conv_1 = tf.keras.layers.Conv2D(
            num_nodes_first_conv_layer,
            kernel_size,
            activation=conv_layer_activation_function,
            padding='same',
            name='conv2d_1'
        )
        flattened_image_size = self._input_image_shape[0] * self._input_image_shape[1]
        if num_nodes_second_conv_layer is not None:
            conv_2 = tf.keras.layers.Conv2D(
                num_nodes_second_conv_layer,
                kernel_size,
                padding='same',
                name='conv2d_2'
            )
            model = tf.keras.Sequential([
                input_layer,
                conv_1,
                conv_2,
                tf.keras.layers.AveragePooling2D(
                    # .. todo:: Try pool (1, INPUT_IMAGE_SHAPE[0]) and strides (1, INPUT_IMAGE_SHAPE[0])
                    pool_size=(1, self._input_image_shape[1]),
                    strides=(1, self._input_image_shape[1]),
                    padding='same',
                    data_format='channels_last'
                ),
                tf.keras.layers.Flatten(),
                output_layer
            ])
        else:
            model = tf.keras.Sequential([
                input_layer,
                conv_1,
                tf.keras.layers.AveragePooling2D(
                    # .. todo: This is pooling along the height dimension; not the width dimension. Should be switched.
                    pool_size=(self._input_image_shape[0], self._input_image_shape[1]),
                    strides=(self._input_image_shape[0], self._input_image_shape[1]), padding='same',
                    data_format='channels_last'
                ),
                tf.keras.layers.Flatten(),
                output_layer
            ])
        model.build(input_shape=(None,) + self._input_image_shape)
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        return model


class FromScratch1DConvHyperModel(PipingDetectorHyperModel):

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]], hyperparameters: Optional[kt.HyperParameters] = None):
        super().__init__(input_image_shape, num_classes, loss, metrics, hyperparameters)
        self._name = 'FromScratch1DConvHyperModel'

    def build(self, hp: kt.HyperParameters, *args, **kwargs) -> Model:
        logger.info(f"Building hyperparameter with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._input_image_shape)
        if self._num_classes == 2:
            # The final dense layer should just be a single neuron with a sigmoid activation function.
            output_layer = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid', name='output_layer')
        else:
            output_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax', name='output_layer')
        kernel_size = hp['kernel_size']
        conv_layer_activation_function = hp['conv_layer_activation_function']
        num_nodes_first_conv_layer = hp['num_nodes_conv_1']
        conv_1 = tf.keras.layers.Conv1D(
            num_nodes_first_conv_layer, kernel_size=kernel_size, activation=conv_layer_activation_function,
            strides=1, padding='same', use_bias=True
        )
        model = tf.keras.Sequential([
            input_layer,
            conv_1,
            # tf.keras.layers.Add(),
            tf.keras.layers.Flatten(name='flatten_1'),
            tf.keras.layers.Dense(self._input_image_shape[0], activation='relu', name='dense_1'),
            output_layer
        ])
        model.build(input_shape=(None,) + self._input_image_shape)
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        return model


class FromScratchBasicHyperModel(PipingDetectorHyperModel):

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]],
            hyperparameters: Optional[kt.HyperParameters] = None):
        super().__init__(input_image_shape, num_classes, loss, metrics, hyperparameters)
        self._name = 'FromScratchBasicHyperModel'

    def build(self, hp: kt.HyperParameters, *args, **kwargs) -> Model:
        logger.info(f"Building hyperparameter with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._input_image_shape)
        if self._num_classes == 2:
            # The final dense layer should just be a single neuron with a sigmoid activation function.
            output_layer = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid', name='output_layer')
        else:
            output_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax', name='output_layer')
        kernel_size = (hp['kernel_size'], hp['kernel_size'])
        conv_layer_activation_function = hp['conv_layer_activation_function']
        model = tf.keras.Sequential([
            input_layer,
            tf.keras.layers.Conv1D()
        ])
        # model = tf.keras.Sequential([
        #     input_layer,
        #     tf.keras.layers.Conv2D(INPUT_IMAGE_SHAPE[0]//4, kernel_size=kernel_size, activation=conv_layer_activation_function, name='conv_1'),
        #     tf.keras.layers.Conv2D(INPUT_IMAGE_SHAPE[0]//16, strides=(2, 2), kernel_size=kernel_size, activation=conv_layer_activation_function, name='conv_2'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_1', padding='same'),
        #     tf.keras.layers.Flatten(name='flatten_1'),
        #     tf.keras.layers.Dense(INPUT_IMAGE_SHAPE[0], activation='relu', name='dense_1'),
        #     output_layer
        # ])
        model.build(input_shape=(None,) + self._input_image_shape)
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        return model


class FromScratchHyperModel(PipingDetectorHyperModel):

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]],
            hyperparameters: Optional[kt.HyperParameters] = None):
        """

        Args:
            num_classes:
            loss:
            metrics:
        """
        super().__init__(input_image_shape, num_classes, loss, metrics, hyperparameters)
        self._name = 'FromScratchHyperModel'

    def build(self, hp: kt.HyperParameters, *args, **kwargs) -> Model:
        """
        Builds the model with the provided hyperparameters. The base model is a CNN model inspired by LeNet-5 from the
        book *Deep Learning Illustrated* by Krohn, Beyleveld, and Bassens.

        Args:
            hp:
            *args:
            **kwargs:

        Returns:

        """
        logger.info(f"Building hypermodel with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._input_image_shape)
        # Images are already individually normalized to range 0 - 1 by the preprocessing and load function.
        # normalization_layer = tf.keras.layers.Normalization(
        #     mean=self._training_set_means, variance=self._training_set_variances, name='normalization_layer'
        # )
        if self._num_classes == 2:
            # The final dense layer should just be a single neuron with a sigmoid activation function.
            output_layer = tf.keras.layers.Dense(self._num_classes - 1, activation='sigmoid', name='output_layer')
        else:
            output_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax', name='output_layer')
        kernel_size = (hp['kernel_size'], hp['kernel_size'])
        conv_layer_activation_function = hp['conv_layer_activation_function']
        model = tf.keras.Sequential([
            input_layer,
            # normalization_layer,
            tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=conv_layer_activation_function,
                                   name='conv_1'),
            tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=conv_layer_activation_function,
                                   name='conv_2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_1'),
            tf.keras.layers.Dropout(0.25, name='dropout_1'),
            tf.keras.layers.Flatten(name='flatten_1'),
            tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.5, name='dropout_2'),
            output_layer
        ])
        model.build(input_shape=(None,) + self._input_image_shape)
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        return model


def hive_name_to_shorthand(full_hive_name: str) -> str:
    """
    Converts a full hive name to a shorthand version of the hive name. For instance, the hive name ``"AppMAIS1L"``
    would be converted to ``"1L"``.

    Args:
        full_hive_name (str): The full hive name to convert to a shorthand version.

    Returns:
        str: The shorthand version of the hive name.
    """
    digit_index = -1
    for i, char in enumerate(full_hive_name):
        if char.isdigit():
            digit_index = i
            break
    return full_hive_name[digit_index:]


class PipingDetector:

    def __init__(
            self, overwrite_existing: bool, train_from_scratch: bool, use_grayscale: bool, stft_params: StftParams,
            appmais_user_name: str, appmais_password: str, hive_names_for_analysis: Optional[List[str]] = None,
            use_gpu: Optional[bool] = True, batch_size: Optional[int] = 64, seed: Optional[int] = 42):
        """
        Main driver class for the piping detection problem.

        Args:
            overwrite_existing (bool): A boolean value indicating if local model output directories should be
              overwritten. For the most part this is not utilized, and crucial output logs are sent to WandB. But this
              flag will play a role for local TensorBoard-driven output logging.
            train_from_scratch (bool): A boolean value indicating if the model should be trained from scratch on the
              training data specified, or simply fine-tuned from the pre-existing VGG19 model weights.
            use_grayscale (bool): A boolean value indicating if training (or optionally fine-tuning) should be performed
              on grayscale images. If this value is ``True`` then the RGB images exported by the database will be
              converted to grayscale prior to training. If this value is ``False`` then the color-mapped RGB images
              directly exported from the database will be utilized for training.
            stft_params (StftParams): An instance of the ``StftParams`` class that specifies the STFT parameters which
              were utilized for the data that was exported from the database. These values are primarily used for
              visualizations.
            hive_names_for_analysis (Optional[List[str]]): A list of valid AppMAIS hive names that will be used for
              training, and inference. If this value is ``None`` then all AppMAIS hive names in the data root directory
              will be leveraged for both training and inference.
            use_gpu (Optional[bool]): A boolean value indicating if the GPU should be utilized for training and
              inference. If no value is provided then the GPU will be utilized if available.
            batch_size (Optional[int]): The batch size to utilize for training and inference. If no value is provided
              then the batch size will default to ``64``.
            seed (Optional[int]): The seed to utilize for training and inference. If no value is provided then the seed
              will default to the standard choice of ``42``.

        """
        self._overwrite_existing = overwrite_existing
        self._train_from_scratch = train_from_scratch
        self._use_grayscale = use_grayscale
        self._hive_names_for_analysis = hive_names_for_analysis
        self._use_gpu = use_gpu
        self._batch_size = batch_size
        self._seed = seed
        self._vgg_19_input_shape = (224, 224)
        logger.warning(f"Downsizing image by half.")
        if self._use_grayscale:
            self._input_shape = (224 // 2, 224 // 2, 1)
        else:
            self._input_shape = (224 // 2, 224 // 2, 3)
        self._stft_params = stft_params
        self._appmais_user_name = appmais_user_name
        self._appmais_password = appmais_password
        self._hyperparameters = kt.HyperParameters()
        # Retrieve list of all valid AppMAIS hives from the database:
        if socket.gethostname() == 'lambda':
            bee_db_mongo_port = 27017
            appmais_server = SSHTunnelForwarder(
                '152.10.10.14',
                ssh_username=self._appmais_user_name,
                ssh_password=self._appmais_password,
                remote_bind_address=('127.0.0.1', bee_db_mongo_port)
            )
            appmais_server.start()
            self._bee_db_mongo_client = MongoClient('127.0.0.1', appmais_server.local_bind_port)
            self._beedb_utils = BeeDBUtils(mongo_client=self._bee_db_mongo_client)
        else:
            raise NotImplementedError("This code was designed to be run on the lambda compute server.")
        self._all_hive_names = self._beedb_utils.get_hive_names()
        if not self._use_gpu:
            # See: https://www.tensorflow.org/api_docs/python/tf/config/set_visible_devices
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                # Disable GPUs:
                tf.config.set_visible_devices([], 'GPU')
                logical_devices = tf.config.list_logical_devices('GPU')
                assert len(logical_devices) == len(physical_devices) - NUM_PHYSICAL_GPUS
            except Exception:
                # Invalid device or cannot modify virtual devices once initialized.
                logger.error(
                    "Failed to disable available physical GPUs. Invalid device or cannot modify virtual devices "
                    "once initialized.")
                pass
        tf.keras.utils.set_random_seed(self._seed)
        # tf.random.set_seed(self._seed)
        # tf.config.experimental.enable_op_determinism()
        # Generate a unique group for this run:
        hive_names_shorthand = [hive_name_to_shorthand(hive_name) for hive_name in self._hive_names_for_analysis]
        self._wab_group_name = f"{''.join(hive_names_shorthand)}-{wandb.util.generate_id()}"
        # TensorBoard logging directory:
        self._tensor_board_log_dir = os.path.abspath(f'/tmp/tb_logs/{self._wab_group_name}')
        if os.path.exists(self._tensor_board_log_dir):
            if self._overwrite_existing:
                # Purge any existing TensorBoard data in this logging directory:
                shutil.rmtree(self._tensor_board_log_dir, ignore_errors=True)
            else:
                raise OSError(f"TensorBoard logging directory already exists, and overwrite_existing is set to "
                              f"{self._overwrite_existing}: {self._tensor_board_log_dir}")
        else:
            # Create the directory tree necessary to write TensorBoard data:
            os.makedirs(self._tensor_board_log_dir, exist_ok=False)
        '''
        Declare model hyperparameters which will remain constant/unmodified during the search process:
        '''
        adam_learning_rate_fixed_hp = self._hyperparameters.Fixed(
            name='adam_learning_rate',
            value=0.001,
            parent_name=None,
            parent_values=None
        )
        # kernel_width_hp = self._hyperparameters.Fixed(
        #     name='kernel_width_size',
        #     value=11,
        #     parent_name=None,
        #     parent_values=None
        # )
        # kernel_height_hp = self._hyperparameters.Fixed(
        #     name='kernel_height_size',
        #     value=11,
        #     parent_name=None,
        #     parent_values=None
        # )
        # num_nodes_conv_1_hp = self._hyperparameters.Fixed(
        #     name='num_nodes_conv_1',
        #     value=2**3,
        #     parent_name=None,
        #     parent_values=None
        # )
        # num_nodes_conv_2_hp = self._hyperparameters.Fixed(
        #     name='num_nodes_conv_2',
        #     value=2**2,
        #     parent_name=None,
        #     parent_values=None
        # )
        conv_layer_activation_function_hp = self._hyperparameters.Fixed(
            name='conv_layer_activation_function',
            value='tanh',
            parent_name=None,
            parent_values=None
        )
        '''
        Declare model hyperparameters which will be tuned/modified during the search process:
        '''
        num_nodes_conv_1_hp = self._hyperparameters.Choice(
            name='num_nodes_conv_1',
            values=[2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3],
            ordered=True,
            default=2 ** 2
        )
        num_nodes_conv_2_hp = self._hyperparameters.Choice(
            name='num_nodes_conv_2',
            values=[2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3],
            ordered=True,
            default=2 ** 0
        )
        # num_thawed_layers_hp = hyperparameters.Choice(
        #     name='num_thawed_layers',
        #     values=[1, 2],
        #     ordered=True,
        #     default=1
        # )
        # adam_learning_rate_hp = hyperparameters.Choice(
        #     name='adam_learning_rate',
        #     # values=[0.1, 0.01, 0.001],
        #     values=[0.001],
        #     ordered=True,
        #     default=0.01
        # )
        # Square kernels in the search space:
        kernel_size = self._hyperparameters.Choice(
            name='kernel_size',
            values=[1, 3, 5, 7, 9, 11],
            ordered=True,
            default=11
        )
        # conv_layer_activation_function_hp = hyperparameters.Choice(
        #     name='conv_layer_activation_function',
        #     values=['relu', 'leaky_relu', 'tanh'],
        #     ordered=False,
        #     default='relu'
        # )
        # num_nodes_conv_1_hp = hyperparameters.Choice(
        #     name='num_nodes_conv_1',
        #     values=[INPUT_IMAGE_SHAPE[0]//8, INPUT_IMAGE_SHAPE[0]//16, INPUT_IMAGE_SHAPE[0]//32],
        #     ordered=True,
        #     default=INPUT_IMAGE_SHAPE[0]//8
        # )
        '''
        Initialize a TensorFlow dataset from disk.
        See Also:
            https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
        '''
        # Initialize TensorFlow datasets from disk:
        data_root_dir = os.path.abspath(f'/home/bee/piping-detection/split_spectra_images')
        train_ds, val_ds, test_ds, resampled_train_steps_per_epoch, resampled_val_steps_per_epoch, _ = (
            self.load_datasets(
                root_data_dir=data_root_dir,
                upload_to_wandb=False,
                hive_names=self._hive_names_for_analysis,
                over_sample_train_set_positive_class=True,
                over_sample_val_set_positive_class=True,
                over_sample_test_set_positive_class=False,
            )
        )
        '''
        Initialize the HyperModel.
        See Also:
            https://keras.io/api/keras_tuner/hypermodels/
        '''
        if self._train_from_scratch:
            ''' Train-from-scratch super-basic Le-Net-5 styled hypermodel from Deep Learning Illustrated '''
            # hyper_model = FromScratchHyperModel(
            #     input_image_shape=self._input_shape,
            #     num_classes=2, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #     # num_classes=2, loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.5, gamma=0.2, from_logits=False),
            #     metrics=[
            #         'binary_accuracy', tf.keras.losses.BinaryCrossentropy(from_logits=False), tf.keras.metrics.TruePositives(),
            #         tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
            #         BalancedBinaryAccuracy()
            #     ],
            #     hyperparameters=self._hyperparameters
            # )
            ''' Train from scratch even simpler model derived from LeNet with ~9M parameters instead of 70M: '''
            # hyper_model = FromScratchBasicHyperModel(
            #     input_image_shape=self._input_shape,
            #     num_classes=2, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #     metrics=[
            #         'binary_accuracy', tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #         tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(),
            #         tf.keras.metrics.FalseNegatives(), BalancedBinaryAccuracy()
            #     ],
            #     hyperparameters=self._hyperparameters
            # )
            '''
            Train from scratch 1D convolutional hypermodel which is designed to represent each frequency bin:
            '''
            # hyper_model = FromScratch1DConvHyperModel(
            #     input_image_shape=self._input_shape,
            #     num_classes=2, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #     metrics=[
            #         'binary_accuracy', tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #         tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(),
            #         tf.keras.metrics.FalseNegatives(), BalancedBinaryAccuracy()
            #     ],
            #     hyperparameters=self._hyperparameters
            # )
            ''' Train from scratch 2D convolutional hypermodel with AveragePooling across time: '''
            self._hyper_model = FromScratch2DConvHyperModel(
                input_image_shape=self._input_shape,
                num_classes=2, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                    'binary_accuracy', tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives(), BalancedBinaryAccuracy()
                ],
                hyperparameters=self._hyperparameters
            )
        else:
            # Transfer learning pre-trained hypermodel.
            self._hyper_model = PipingDetectorHyperModel(
                input_image_shape=self._input_shape,
                num_classes=2, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['binary_accuracy', 'MSE', 'binary_crossentropy'],
                hyperparameters=self._hyperparameters
            )
        '''
        Initialize the Oracle.

        See Also:
            https://keras.io/api/keras_tuner/oracles/hyperband/
        '''
        self._oracle = kt.oracles.HyperbandOracle(
            objective=kt.Objective(name='val_loss', direction='min'),
            # .. todo: Set this slightly higher than the number of epochs needed to train a single model:
            max_epochs=100,
            # Reduction factor for the number of epochs  and number of models for each bracket.
            factor=3,
            # .. todo:: Up this count as high as computational budget will allow for:
            hyperband_iterations=1,
            seed=SEED,
            hyperparameters=self._hyper_model.hyperparameters,
            tune_new_entries=True,
            allow_new_entries=True,
            max_retries_per_trial=0,
            max_consecutive_failed_trials=3,
            # overwrite=OVERWRITE_EXISTING,
            # directory=DATA_TEMP_DIR
        )
        '''
        Initialize the Tuner.

        See Also:
            https://keras.io/api/keras_tuner/tuners/
        '''
        self._tuner = WBTuner(
            hypermodel=self._hyper_model,
            seed=self._seed,
            oracle=self._oracle,
            wab_group_name=self._wab_group_name,
            directory=DATA_TEMP_DIR,
            overwrite=self._overwrite_existing
        )
        ''' Callbacks for: monitoring training, early stopping, debugging, etc.  '''
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=tensor_board_log_dir,
        #     update_freq='epoch',
        #     profile_batch=(10, 15)
        # )
        ''' Sample a fixed set of images from the training set to run Grad-CAM on during training: '''
        # num_images_to_map = 5
        # dataset = dataset.unbatch()
        # random_batch = dataset.take(num_images_to_map)
        # random_images = np.zeros((num_images_to_map, *image_shape))
        # random_labels = np.zeros((num_images_to_map, 1))
        # for i, (image, label) in enumerate(random_batch):
        #     random_images[i] = image.numpy()
        #     random_labels[i] = label.numpy()
        # Run the hyperparameter search with early stopping based on the validation set:
        # tuner.search(train_ds, epochs=1, validation_data=val_ds, callbacks=[stop_early, tensorboard_callback])
        self._tuner.search(
            train_ds, steps_per_epoch=resampled_train_steps_per_epoch,
            validation_steps=resampled_val_steps_per_epoch, epochs=400, validation_data=val_ds, callbacks=[stop_early]
        )
        # Get the optimal hyperparameters
        best_hps = self._tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters: {best_hps.values}")
        # Create a new WANDB run for the model to fit with optimal hyperparameters:
        trial_run = wab.init(
            project="PipingDetection",
            config=best_hps,
            group=self._wab_group_name,
            save_code=True
        )
        history = self._tuner.hypermodel.fit(
            model=self._tuner.hypermodel.build(best_hps, input_shape=self._input_shape),
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=30,
            callbacks=[stop_early],
            steps=resampled_train_steps_per_epoch,
            val_steps=resampled_val_steps_per_epoch
        )
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        logger.info(f"Best epoch with optimal hyperparameter settings: {best_epoch}")
        trial_run.finish()

    @staticmethod
    @tf.function
    def __preprocess_image_path(image_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        This helper method is intended to be mapped over an existing :class:`tf.data.Dataset` object which contains only
        image file paths. The ``label`` argument is intended to be the same for all provided ``image_path`` values, and may
        be applied to the :class:`tf.data.Dataset` via the :meth:`tf.data.Dataset.map` method and the
        :meth:`functools.partial` wrapper function. For instance::
            ```
            train_ds_neg_files = tf.data.Dataset.list_files(negative_train_data_root_dir + '/*.png')
            label = tf.constant(0, dtype=tf.int32)
            train_ds_neg = train_ds_neg_files.map(partial(__preprocess_image_path, label=label))
            ```
        Or alternatively::
            ```
            train_ds_pos_files = tf.data.Dataset.list_files(positive_train_data_root_dir + '/*.png')
            label = tf.constant(1, dtype=tf.int32)
            train_ds_pos = train_ds_pos_files.map(partial(__preprocess_image_path, label=label))
            ```
        This helper method should only be needed in the case where the user wishes to weight the binary classification
        dataset (e.g. in the case of an unbalanced dataset). If the user does not wish to weight the binary classification
        dataset, then simply using the method :meth:`tf.keras.preprocessing.image_dataset_from_directory` will be easier
        and more efficient.

        Args:
            image_path (str?): A string tensor which houses the path to the image file to which preprocessing is to be
              applied.
            label (tf.Constant): A constant tensor which houses the label to be applied to the image file. This label will
                 be the same for all image files in the dataset when this function is mapped directly to a
                 :class:`tf.data.Dataset`. It is anticipated that a value of ``0`` will be used for negative examples, and
                 a value of ``1`` will be used for positive examples.

        See Also:
            https://www.tensorflow.org/guide/data#consuming_sets_of_files
            https://github.com/keras-team/keras/issues/17141

        Notes:
            We must use this method because the documentation for
            ``tf.keras.preprocessing.image_dataset_from_directory`` is blatantly wrong. When passing in a hardcoded
            list of labels, the directory structure *is not* ignored (as the documentation states it should be). See the
            referenced GitHub issue for more details. Also note that the ``tf.function`` decorator is necessary here for
            efficiency’s sake, you can read more about this decorator in the TensorFlow documentation.

        Returns:
            [tf.Tensor, tf.Tensor] A size-2 tuple containing:
                - **image** (*tf.Tensor*): A ``tf.float32`` tensor which houses the loaded raw image data.
                - **label** (*tf.Tensor*): A ``tf.int32`` tensor which houses the label to be applied to the image file.

        """
        # VGG19_INPUT_SHAPE = (224, 224)
        # Convert from the 'viridis' color map that the images were saved in, back to floating point representation:
        # cm = np.array(plt.get_cmap('viridis', lut=256).colors)
        # cm = cm[:, :3]
        # from scipy.spatial.distance import cdist
        # d = cdist(cm, img.reshape(-1, 3))
        # idxs = np.argmin(d, axis=0)
        # idx = idx / 255
        # idx = idx.reshape((224, 224))

        raw_byte_string = tf.io.read_file(image_path)
        image = tf.image.decode_png(raw_byte_string, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        downsampled_image_size = (VGG19_INPUT_SHAPE[0] // 2, VGG19_INPUT_SHAPE[1] // 2)
        image = tf.image.resize(image, downsampled_image_size)
        image.set_shape(downsampled_image_size + (1,))
        return image, label

    @staticmethod
    def load_datasets_for_hive(dataset_split: DatasetSplit, hive_name: str, hive_dataset_split_root_dir: str, seed: int) \
            -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        .. todo:: Docstrings.

        Args:
            dataset_split:
            hive_name (str):
            hive_dataset_split_root_dir:

        Returns:

        """
        logger.debug(f"Loading {dataset_split} dataset for hive {hive_name}...")
        assert os.path.exists(hive_dataset_split_root_dir), (f"{dataset_split} root directory "
                                                             f"{hive_dataset_split_root_dir} for hive {hive_name} does "
                                                             f"not exist.")
        # Negative sample dataset split directory for the hive:
        negative_split_data_dir = os.path.join(hive_dataset_split_root_dir, 'IsNotPiping')
        negative_split_data_dir_path = Path(negative_split_data_dir)
        assert os.path.exists(negative_split_data_dir), (f"Negative sample {dataset_split} data directory "
                                                         f"{negative_split_data_dir} for hive {hive_name} does not "
                                                         f"exist.")
        num_negative_split_samples = sum(1 for path in negative_split_data_dir_path.rglob('*') if path.is_file())
        logger.debug(f"Identified {num_negative_split_samples} negative {dataset_split} samples for hive {hive_name}.")
        # Positive sample dataset split directory for the hive:
        positive_split_data_dir = os.path.join(hive_dataset_split_root_dir, 'IsPiping')
        positive_split_data_dir_path = Path(positive_split_data_dir)
        assert os.path.exists(positive_split_data_dir), (f"Positive sample {dataset_split} data directory "
                                                         f"{positive_split_data_dir} for hive {hive_name} does not "
                                                         f"exist.")
        num_positive_split_samples = sum(1 for path in positive_split_data_dir_path.rglob('*') if path.is_file())
        logger.debug(f"Identified {num_positive_split_samples} positive {dataset_split} samples for hive {hive_name}.")
        # Load negative label split data:
        split_ds_neg_files = tf.data.Dataset.list_files(negative_split_data_dir + '/*.png', seed=seed)
        # Remove half of the training data:
        if dataset_split == DatasetSplit.TRAIN:
            logger.warning(f"Removing half of the negative {dataset_split} data for hive {hive_name}.")
            split_ds_neg_files = split_ds_neg_files.take(num_negative_split_samples // 2)
            logger.debug(
                f"Identified {split_ds_neg_files.cardinality().numpy()} negative {dataset_split} samples for hive {hive_name}.")
        label = tf.constant(0, dtype=tf.int32)
        split_ds_neg = split_ds_neg_files.map(partial(PipingDetector.__preprocess_image_path, label=label))
        # Load positive label split data:
        split_ds_pos_files = tf.data.Dataset.list_files(positive_split_data_dir + '/*.png', seed=seed)
        # Remove half of the training data:
        if dataset_split == DatasetSplit.TRAIN:
            logger.warning(f"Removing half of the positive {dataset_split} data for hive {hive_name}.")
            split_ds_pos_files = split_ds_pos_files.take(num_positive_split_samples // 2)
            logger.debug(
                f"Identified {split_ds_pos_files.cardinality().numpy()} positive {dataset_split} samples for hive {hive_name}.")
        label = tf.constant(1, dtype=tf.int32)
        split_ds_pos = split_ds_pos_files.map(partial(PipingDetector.__preprocess_image_path, label=label))
        # Construct unmodified dataset of (image, label) pairs for summary statistics calculations:
        # split_ds = tf.data.Dataset.sample_from_datasets(
        #     datasets=[split_ds_pos, split_ds_neg], weights=[1.0, 1.0], seed=SEED
        # )

        # def __normalize_input(image, label):
        #     return image / 255.0, label

        # num_samples = sum(1 for _ in split_ds)
        # split_ds = split_ds.map(lambda image, label: __normalize_input(image, label))
        # split_ds = split_ds.map(__normalize_input)

        # sums = tf.Variable(initial_value=tf.zeros((3,), dtype=tf.float32), trainable=False, name='sums')
        # split_ds.map(lambda image, label: sums.assign_add(tf.reduce_sum(image, axis=(0, 1))))

        # split_ds = split_ds.map(lambda image, label: )

        # sums = tf.zeros((3,), dtype=tf.float32)
        # split_ds.map(lambda image, label: tf.reduce_sum(image, axis=(0, 1)) + sums)
        # means = tf.math.divide(sums, num_samples)
        # tf.math.d
        # means = sums / num_samples
        # means = tf.math.reduce_mean(split_ds.as_numpy_iterator(), axis=-1)

        # def f(x, y):
        #     print(x)
        #     print(y)
        #     return x + y[1]
        # split_ds.reduce(np.zeros_like((224, 224, 3)), f)
        # for image, label in split_ds:
        #     mean.update_state(image[..., -1])
        # tfp.stats.variance(split_ds.map(lambda image, label: image))

        # split_ds_means = tf.reduce_mean(split_ds.map(lambda image, label: image), axis=-1)
        # split_ds_variances = tf.math.reduce_variance(split_ds.map(lambda image, label: image), axis=-1)
        return split_ds_neg_files, split_ds_neg, split_ds_pos_files, split_ds_pos

    @staticmethod
    def aggregate_datasets_across_hives(hive_names: List[str], root_data_dir: str, seed: int):
        train_ds_neg_files, train_ds_neg, train_ds_pos_files, train_ds_pos = None, None, None, None
        val_ds_neg_files, val_ds_neg, val_ds_pos_files, val_ds_pos = None, None, None, None
        test_ds_neg_files, test_ds_neg, test_ds_pos_files, test_ds_pos = None, None, None, None
        logger.debug(f"Aggregating pre-split data for hives {hive_names}.")
        # First load each hive's individual (probably unbalanced) datasets:
        for i, hive_name in enumerate(hive_names):
            hive_data_dir = os.path.join(root_data_dir, hive_name)
            assert os.path.exists(
                hive_data_dir), f"Could not find data directory for hive {hive_name} at {hive_data_dir}"
            # Training data for this hive:
            hive_dataset_train_dir = os.path.join(hive_data_dir, 'train')
            hive_train_ds_neg_files, hive_train_ds_neg, hive_train_ds_pos_files, hive_train_ds_pos = (
                PipingDetector.load_datasets_for_hive(
                    dataset_split=DatasetSplit.TRAIN,
                    hive_name=hive_name,
                    hive_dataset_split_root_dir=hive_dataset_train_dir,
                    seed=seed
                )
            )
            if i == 0:
                train_ds_neg_files = hive_train_ds_neg_files
                train_ds_neg = hive_train_ds_neg
                train_ds_pos_files = hive_train_ds_pos_files
                train_ds_pos = hive_train_ds_pos
            else:
                train_ds_neg_files = train_ds_neg_files.concatenate(hive_train_ds_neg_files, name='train_ds_neg_files')
                train_ds_neg = train_ds_neg.concatenate(hive_train_ds_neg, name='train_ds_neg')
                train_ds_pos_files = train_ds_pos_files.concatenate(hive_train_ds_pos_files, name='train_ds_pos_files')
                train_ds_pos = train_ds_pos.concatenate(hive_train_ds_pos, name='train_ds_pos')
            # Validation data for this hive:
            hive_dataset_val_dir = os.path.join(hive_data_dir, 'val')
            hive_val_ds_neg_files, hive_val_ds_neg, hive_val_ds_pos_files, hive_val_ds_pos = (
                PipingDetector.load_datasets_for_hive(
                    dataset_split=DatasetSplit.VALIDATION,
                    hive_name=hive_name,
                    hive_dataset_split_root_dir=hive_dataset_val_dir,
                    seed=seed
                )
            )
            if i == 0:
                val_ds_neg_files = hive_val_ds_neg_files
                val_ds_neg = hive_val_ds_neg
                val_ds_pos_files = hive_val_ds_pos_files
                val_ds_pos = hive_val_ds_pos
            else:
                val_ds_neg_files = val_ds_neg_files.concatenate(hive_val_ds_neg_files, name='val_ds_neg_files')
                val_ds_neg = val_ds_neg.concatenate(hive_val_ds_neg, name='val_ds_neg')
                val_ds_pos_files = val_ds_pos_files.concatenate(hive_val_ds_pos_files, name='val_ds_pos_files')
                val_ds_pos = val_ds_pos.concatenate(hive_val_ds_pos, name='val_ds_pos')
            # Testing data for this hive:
            hive_dataset_test_dir = os.path.join(hive_data_dir, 'test')
            hive_test_ds_neg_files, hive_test_ds_neg, hive_test_ds_pos_files, hive_test_ds_pos = (
                PipingDetector.load_datasets_for_hive(
                    dataset_split=DatasetSplit.TEST,
                    hive_name=hive_name,
                    hive_dataset_split_root_dir=hive_dataset_test_dir,
                    seed=seed
                )
            )
            if i == 0:
                test_ds_neg_files = hive_test_ds_neg_files
                test_ds_neg = hive_test_ds_neg
                test_ds_pos_files = hive_test_ds_pos_files
                test_ds_pos = hive_test_ds_pos
            else:
                test_ds_neg_files = test_ds_neg_files.concatenate(hive_test_ds_neg_files, name='test_ds_neg_files')
                test_ds_neg = test_ds_neg.concatenate(hive_test_ds_neg, name='test_ds_neg')
                test_ds_pos_files = test_ds_pos_files.concatenate(hive_test_ds_pos_files, name='test_ds_pos_files')
                test_ds_pos = test_ds_pos.concatenate(hive_test_ds_pos, name='test_ds_pos')
        # # Now combine all hives' training datasets into a single dataset:
        # train_ds_neg_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_train_ds_neg_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # train_ds_neg = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_train_ds_neg,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # train_ds_pos_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_train_ds_pos_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # train_ds_pos = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_train_ds_pos,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # # Combine all hive's validation datasets into a single dataset:
        # val_ds_neg_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_val_ds_neg_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # val_ds_neg = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_val_ds_neg,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # val_ds_pos_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_val_ds_pos_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # val_ds_pos = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_val_ds_pos,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # # Combine all hive's testing datasets into a single dataset:
        # test_ds_neg_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_test_ds_neg_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # test_ds_neg = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_test_ds_neg,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # test_ds_pos_files = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_test_ds_pos_files,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        # test_ds_pos = tf.data.Dataset.sample_from_datasets(
        #     datasets=hives_test_ds_pos,
        #     weights=[1.0, 1.0],
        #     seed=seed,
        #     stop_on_empty_dataset=False
        # )
        return (train_ds_neg_files, train_ds_neg, train_ds_pos_files, train_ds_pos, val_ds_neg_files, val_ds_neg,
                val_ds_pos_files, val_ds_pos, test_ds_neg_files, test_ds_neg, test_ds_pos_files, test_ds_pos)

    def get_oversampled_dataset(
            self, dataset_split: DatasetSplit, split_ds_neg: tf.data.Dataset, split_ds_pos: tf.data.Dataset,
            split_ds_neg_files: tf.data.Dataset, split_ds_pos_files: tf.data.Dataset):
        """
        Takes as input an unbalanced dataset split (train/val/test data) and over-samples the positive class label so
        that the prevalence of samples is 50/50 among positive and negative classes.

        Args:
            dataset_split (DatasetSplit): A :class:`DatasetSplit` enum value indicating which dataset split (e.g.
              train/val/test) is currently being oversampled.
            split_ds_neg (tf.data.Dataset): A ``tf.data.Dataset`` object which houses the negative class unbalanced
              sample data for a particular dataset split (e.g. ``TRAIN``/``VAL``/``TEST``). This dataset is assumed to
              contain the images themselves.
            split_ds_pos (tf.data.Dataset): A ``tf.data.Dataset`` object which houses the positive class unbalanced
              sample data for a particular dataset split (e.g. ``TRAIN``/``VAL``/``TEST``). This dataset is assumed to
              contain the images themselves.
            split_ds_neg_files (tf.data.Dataset): A ``tf.data.Dataset`` object which houses the negative class
              unbalanced file paths for a particular dataset split (e.g. ``TRAIN``/``VAL``/``TEST``). This dataset is
              assumed to contain only the paths to the images, which themselves held in the ``split_ds_neg`` dataset.
            split_ds_pos_files (tf.data.Dataset): A ``tf.data.Dataset`` object which houses the positive class
              unbalanced file paths for a particular dataset split (e.g. ``TRAIN``/``VAL``/``TEST``). This dataset is
              assumed to contain only the paths to the images, which themselves held in the ``split_ds_pos`` dataset.
            summary_statistics (bool): A boolean indicating if summary statistics should be computed for the provided
              data. This argument is primarily useful for retrieving the mean and variance of the training data, which
              can then be used to normalize the validation and testing data.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, int, tf.Tensor, tf.Tensor]: A size-5 tuple containing:
              - **split_ds_files** (*tf.data.Dataset*): A ``tf.data.Dataset`` object which houses the loaded and
                weighted dataset's file paths. This dataset is batched and shuffled in the same order/permutation as
                ``split_ds``.
              - **split_ds** (*tf.data.Dataset*): A ``tf.data.Dataset`` object which houses the loaded and weighted
                dataset. The dataset will be weighted according to the provided ``positive_class_weighting`` and
                ``negative_class_weighting`` values. This dataset will be shuffled and batched in the same
                order/permutation as ``split_ds_files``.
              - **resampled_steps_per_epoch** (*int*): The number of steps per epoch that would be required to see each
                negative sample at least once during training. Since it is assumed that the most prevalent class label
                is the positive class, this value instructs as to how many steps per epoch are required to be performed
                to see every element/sample of the majority class.

        Notes:
            The negative class here is the "IsNotPiping" class, and the positive class is the "IsPiping" class. The
            negative class is the majority class, and the positive class is the minority class. The datasets here are
            set up so that each epoch should run for ``resampled_steps_per_epoch`` steps, allowing each epoch to see
            every negative sample at least once. Every positive sample will be repeated (e.g. seen multiple times
            per-epoch) due to the class imbalance. In essence, this is hence oversampling the less prevalent positive
            class.

        """
        # Get the number of negative samples in the dataset split:
        num_negative_split_samples = 0
        for i, _ in enumerate(split_ds_neg):
            num_negative_split_samples += 1
        logger.debug(f"Detected {num_negative_split_samples} negative samples in the unbalanced {dataset_split} "
                     f"dataset.")
        # Get the number of positive samples in the dataset split:
        num_positive_split_samples = 0
        for i, _ in enumerate(split_ds_pos):
            num_positive_split_samples += 1
        logger.debug(f"Detected {num_positive_split_samples} positive samples in the unbalanced {dataset_split} "
                     f"dataset.")
        # Construct Oversampled dataset of (image, label) pairs:
        split_ds_neg = split_ds_neg.shuffle(
            buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
        ).repeat()
        split_ds_pos = split_ds_pos.shuffle(
            buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
        ).repeat()
        split_ds = tf.data.Dataset.sample_from_datasets(
            datasets=[split_ds_pos, split_ds_neg],
            weights=[0.5, 0.5],
            seed=self._seed
        )
        # .. todo:: The 2 here is hardcoded because we are doing a 50/50 split of positive and negative examples.
        #     This would need to be changed if we were to do a different split.
        # .. todo:: The resampled_steps_per_epoch should be dependent on the majority class label, not necessarily only
        #     the negative class.
        resampled_steps_per_epoch = np.ceil(num_negative_split_samples / (BATCH_SIZE / 2))
        # Construct oversampled dataset of files:
        split_ds_neg_files = split_ds_neg_files.shuffle(
            buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
        ).repeat()
        split_ds_pos_files = split_ds_pos_files.shuffle(
            buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
        ).repeat()
        split_ds_files = tf.data.Dataset.sample_from_datasets(
            datasets=[split_ds_pos_files, split_ds_neg_files],
            weights=[0.5, 0.5],
            seed=self._seed
        )
        # Shuffle and pre-batch datasets:
        split_ds = split_ds.shuffle(buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False)
        split_ds = split_ds.batch(batch_size=self._batch_size, drop_remainder=True)
        # split_ds = split_ds.cache()
        split_ds_files = split_ds_files.shuffle(
            buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
        )
        split_ds_files = split_ds_files.batch(batch_size=self._batch_size, drop_remainder=True)
        # split_ds_files = train_ds_files.cache()
        return split_ds_files, split_ds, resampled_steps_per_epoch

    def load_datasets(
            self, root_data_dir: str, upload_to_wandb: bool, hive_names: Optional[List[str]] = None,
            over_sample_train_set_positive_class: bool = True, over_sample_val_set_positive_class: bool = False,
            over_sample_test_set_positive_class: bool = False) \
            -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int], Optional[int]]:
        """
        This method handles the loading and weighting of the training, validation, and testing datasets. Data will be
        loaded from the hives specified in the list of provided ``hives_names``. If no ``hive_names`` are provided then
        the data from all hives will be leveraged.

        Args:
            root_data_dir (str): The root directory where the training data images are located.
            upload_to_wandb (bool): Indicates whether training data should be uploaded directly to WandB. This may slow
              down :class:`tf.data.Dataset` loading, but can be helpful for debugging.
            hive_names (Optional[List[str]]): A list of hive names to load data for. If None, all hives will be loaded.
            over_sample_train_set_positive_class (bool): Whether to over sample the positive class in the training
              set. Defaults to True. This parameter is useful for highly imbalanced binary classification datasets.
            over_sample_val_set_positive_class (bool): Whether to over sample the positive class in the validation
              set. Note that this parameter defaults to ``False``. This is because it is not recommended to over sample
              the validation set (as it should be representative of the actual data distribution at inference time).
            over_sample_test_set_positive_class (bool): Whether to over sample the positive class in the test set.
              Note that this parameter defaults to ``False``. This is because it is not recommended to over sample the
              testing set (as it should be representative of the actual data distribution at inference time).

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Optional[int]]: A size-4 tuple containing:
            - **train_ds** (tf.data.Dataset): A :class:`tf.data.Dataset` containing the training data. If
              ``over_sample_train_set_positive_class`` is set to ``True``, then it is guaranteed that the number of
              positive and negative examples per-batch will be equal, as the less prevalent positive class will be
              over-sampled. If this is the case, then ``resampled_steps_per_epoch`` will be set to the number of steps
              required to iterate over every negative example in the dataset at least once; and should be passed
              directly to the utilizing model's ``fit`` method.
            - **val_ds** (tf.data.Dataset): A :class:`tf.data.Dataset` containing the validation data. If
              ``over_sample_val_set_positive_class`` is set to ``True``, then it is guaranteed that the number of
              positive and negative examples per-batch will be equal, as the less prevalent positive class will be
              over-sampled. If this is the case, then ``resampled_steps_per_epoch`` will be set to the number of steps
              required to iterate over every negative example in the dataset at least once; and should be passed
              directly to the utilizing model's ``fit`` method. Note that is ill-advised to oversample the validation
              dataset, as it should be representative of the actual data distribution at inference time.
            - **test_ds** (tf.data.Dataset): A :class:`tf.data.Dataset` containing the testing data. If
              ``over_sample_test_set_positive_class`` is set to ``True``, then it is guaranteed that the number of
              positive and negative examples per-batch will be equal, as the less prevalent positive class will be
              over-sampled. If this is the case, then ``resampled_steps_per_epoch`` will be set to the number of steps
              required to iterate over every negative example in the dataset at least once; and should be passed
              directly to the utilizing model's ``fit`` method. Note that it is ill-advised to oversample the testing
              dataset, as it should be representative of the actual data distribution at inference time.
            - **resampled_train_steps_per_epoch** (Optional[int]): The number of steps to take before the training epoch
              is considered finished. This will be ``None`` if ``over_sample_train_set_positive_class`` is ``False``.
              Otherwise, this will be the number of steps required to sample from the training set so that each element
              in the majority/negative class is seen at least once. Note that since the minority/positive class is
              oversampled, then each element in that class will be seen multiple times.
            - **resampled_val_steps_per_epoch** (Optional[int]): The number of steps to take before the validation epoch
              is considered finished. This will be ``None`` if ``over_sample_val_set_positive_class`` is ``False``.
              Otherwise, this will be the number of steps required to sample from the validation set so that each
              element in the majority/negative class is seen at least once. Note that since the minority/positive class
              is oversampled, then each element in that class will be seen multiple times.
            - **resampled_test_steps_per_epoch** (Optional[int]): The number of steps to take before the testing epoch
              is considered finished. This will be ``None`` if ``over_sample_test_set_positive_class`` is ``False``.
              Otherwise, this will be the number of steps required to sample from the testing set so that each element
              in the majority/negative class is seen at least once. Note that since the minority/positive class is
              oversampled, then each element in that class will be seen multiple times.

            - **train_ds_means** (tf.Tensor): A :class:`tf.Tensor` of dtype ``tf.float32`` containing the mean pixel
              values for each RGB channel in the raw training dataset. This value can later be used to normalize the
              validation and testing datasets.
            - **train_ds_variances** (tf.Tensor): A :class:`tf.Tensor` of dtype ``tf.float32`` containing the variance
              of the pixel values for each RGB channel in the raw training dataset. This value can later be used to
              normalize the validation and testing datasets.

        See Also:
           https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#using_tfdata

        """
        assert os.path.exists(root_data_dir)

        # .. todo:: Adapt this to handle multiple hives? Do we upload at the hive level or only the globally split ds?
        if upload_to_wandb:
            raise NotImplementedError('Uploading split datasets across multiple hives to WandB is not yet supported.')
            # upload_split_datasets_to_wandb(
            #     train_data_root_dir=train_data_root_dir,
            #     val_data_root_dir=val_data_root_dir,
            #     test_data_root_dir=test_data_root_dir,
            #     upload_test_data=False
            # )

        (train_ds_neg_files, train_ds_neg, train_ds_pos_files, train_ds_pos, val_ds_neg_files, val_ds_neg,
         val_ds_pos_files, val_ds_pos, test_ds_neg_files, test_ds_neg, test_ds_pos_files, test_ds_pos) = (
            PipingDetector.aggregate_datasets_across_hives(
                hive_names=hive_names,
                root_data_dir=root_data_dir,
                seed=self._seed
            )
        )
        # .. todo:: Reduce the number of samples in the training dataset by half:
        if over_sample_train_set_positive_class:
            train_ds_files, train_ds, resampled_train_steps_per_epoch = self.get_oversampled_dataset(
                dataset_split=DatasetSplit.TRAIN,
                split_ds_neg=train_ds_neg,
                split_ds_pos=train_ds_pos,
                split_ds_neg_files=train_ds_neg_files,
                split_ds_pos_files=train_ds_pos_files
            )
        else:
            train_ds = tf.data.Dataset.concatenate(train_ds_pos, train_ds_neg)
            train_ds = train_ds.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            train_ds = train_ds.batch(batch_size=self._batch_size, drop_remainder=True)
            # train_ds = train_ds.cache()
            train_ds_files = tf.data.Dataset.concatenate(train_ds_pos_files, train_ds_neg_files)
            train_ds_files = train_ds_files.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            train_ds_files = train_ds_files.batch(batch_size=self._batch_size, drop_remainder=True)
            # train_ds_files = train_ds_files.cache()
            resampled_train_steps_per_epoch = None
        # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        ''' Load the validation data: '''
        if over_sample_val_set_positive_class:
            val_ds_files, val_ds, resampled_val_steps_per_epoch = self.get_oversampled_dataset(
                dataset_split=DatasetSplit.VALIDATION,
                split_ds_neg=val_ds_neg,
                split_ds_pos=val_ds_pos,
                split_ds_neg_files=val_ds_neg_files,
                split_ds_pos_files=val_ds_pos_files
            )
        else:
            val_ds = tf.data.Dataset.concatenate(val_ds_pos, val_ds_neg)
            val_ds = val_ds.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            val_ds = val_ds.batch(batch_size=self._batch_size, drop_remainder=True)
            # val_ds = val_ds.cache()
            val_ds_files = tf.data.Dataset.concatenate(val_ds_pos_files, val_ds_neg_files)
            val_ds_files = val_ds_files.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            val_ds_files = val_ds_files.batch(batch_size=self._batch_size, drop_remainder=True)
            # val_ds_files = val_ds_files.cache()
            resampled_val_steps_per_epoch = None
        # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        # val_ds_files = val_ds_files.prefetch(buffer_size=tf.data.AUTOTUNE)
        ''' Load the testing data: '''
        if over_sample_test_set_positive_class:
            test_ds_files, test_ds, resampled_test_steps_per_epoch = self.get_oversampled_dataset(
                dataset_split=DatasetSplit.TEST,
                split_ds_neg=test_ds_neg,
                split_ds_pos=test_ds_pos,
                split_ds_neg_files=test_ds_neg_files,
                split_ds_pos_files=test_ds_pos_files
            )
        else:
            test_ds = tf.data.Dataset.concatenate(test_ds_pos, test_ds_neg)
            test_ds = test_ds.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            test_ds = test_ds.batch(batch_size=self._batch_size, drop_remainder=True)
            # test_ds = test_ds.cache()
            test_ds_files = tf.data.Dataset.concatenate(test_ds_pos_files, test_ds_neg_files)
            test_ds_files = test_ds_files.shuffle(
                buffer_size=self._batch_size, seed=self._seed, reshuffle_each_iteration=False
            )
            test_ds_files = test_ds_files.batch(batch_size=self._batch_size, drop_remainder=True)
            # test_ds_files = test_ds_files.cache()
            resampled_test_steps_per_epoch = None
        # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return (train_ds, val_ds, test_ds, resampled_train_steps_per_epoch, resampled_val_steps_per_epoch,
                resampled_test_steps_per_epoch)

    def upload_split_datasets_to_wandb(
            self, train_data_root_dir: str, val_data_root_dir: str, test_data_root_dir: str,
            upload_test_data: Optional[bool] = False):
        """
        .. todo:: Docstrings.

        Args:
            train_data_root_dir:
            val_data_root_dir:
            test_data_root_dir:
            upload_test_data:

        Returns:

        """
        logger.warning("Uploading raw data to WANDB so this may take a while. Invoke this method with "
                       "`upload_to_wandb=False` to skip this step.")
        negative_train_data_dir = os.path.join(train_data_root_dir, 'IsNotPiping')
        positive_train_data_dir = os.path.join(train_data_root_dir, 'IsPiping')
        negative_val_data_dir = os.path.join(val_data_root_dir, 'IsNotPiping')
        positive_val_data_dir = os.path.join(val_data_root_dir, 'IsPiping')
        negative_test_data_dir = os.path.join(test_data_root_dir, 'IsNotPiping')
        positive_test_data_dir = os.path.join(test_data_root_dir, 'IsPiping')
        negative_label = tf.constant(0, dtype=tf.int32)
        positive_label = tf.constant(1, dtype=tf.int32)
        negative_train_data_files = tf.data.Dataset.list_files(negative_train_data_dir + '/*.png')
        negative_train_data = negative_train_data_files.map(partial(self.__preprocess_image_path, label=negative_label))
        positive_train_data_files = tf.data.Dataset.list_files(positive_train_data_dir + '/*.png')
        positive_train_data = positive_train_data_files.map(partial(self.__preprocess_image_path, label=positive_label))
        train_ds = tf.data.Dataset.sample_from_datasets(
            datasets=[positive_train_data, negative_train_data], weights=[1.0, 1.0], seed=SEED
        )
        train_ds_files = tf.data.Dataset.sample_from_datasets(
            datasets=[positive_train_data_files, negative_train_data_files], weights=[1.0, 1.0], seed=SEED
        )
        train_ds.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
        # train_ds.cache()
        train_ds_files.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
        # train_ds_files.cache()
        num_train_samples = 0
        for _ in train_ds:
            num_train_samples += 1
        negative_val_data_files = tf.data.Dataset.list_files(negative_val_data_dir + '/*.png')
        negative_val_data = negative_val_data_files.map(partial(self.__preprocess_image_path, label=negative_label))
        positive_val_data_files = tf.data.Dataset.list_files(positive_val_data_dir + '/*.png')
        positive_val_data = positive_val_data_files.map(partial(self.__preprocess_image_path, label=positive_label))
        val_ds = tf.data.Dataset.sample_from_datasets(
            datasets=[positive_val_data, negative_val_data], weights=[1.0, 1.0], seed=SEED
        )
        val_ds_files = tf.data.Dataset.sample_from_datasets(
            datasets=[positive_val_data_files, negative_val_data_files], weights=[1.0, 1.0], seed=SEED
        )
        val_ds.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
        # val_ds.cache()
        val_ds_files.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
        # val_ds_files.cache()
        num_val_samples = 0
        for _ in val_ds:
            num_val_samples += 1
        if upload_test_data:
            negative_test_data_files = tf.data.Dataset.list_files(negative_test_data_dir + '/*.png')
            negative_test_data = negative_test_data_files.map(
                partial(self.__preprocess_image_path, label=negative_label))
            positive_test_data_files = tf.data.Dataset.list_files(positive_test_data_dir + '/*.png')
            positive_test_data = positive_test_data_files.map(
                partial(self.__preprocess_image_path, label=positive_label))
            test_ds = tf.data.Dataset.sample_from_datasets(
                datasets=[positive_test_data, negative_test_data], weights=[1.0, 1.0], seed=SEED
            )
            test_ds_files = tf.data.Dataset.sample_from_datasets(
                datasets=[positive_test_data_files, negative_test_data_files], weights=[1.0, 1.0], seed=SEED
            )
            test_ds.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
            # test_ds.cache()
            test_ds_files.shuffle(buffer_size=BATCH_SIZE, seed=SEED, reshuffle_each_iteration=False)
            # test_ds_files.cache()
            num_test_samples = 0
            for _ in test_ds:
                num_test_samples += 1
            num_samples = num_train_samples + num_val_samples + num_test_samples
        else:
            num_samples = num_train_samples + num_val_samples
        upload_split_data_run = wandb.init(
            project="PipingDetection", group=self._wab_group_name, job_type="data_split"
        )
        hive_data_artifact_name = f"{self._wab_group_name}_split_data_{num_samples}"
        # Find the most recent "latest" version fo the full raw data:
        # hive_data_artifact = upload_split_data_run.use_artifact(hive_data_artifact_name + ':latest')
        hive_data_artifact = wandb.Artifact(name=hive_data_artifact_name, type="balanced_split_data")
        # .. todo:: Sync version/artifacts of the dataset by reference.
        # split_data_for_hive_artifact = wandb.Artifact(name=, type="split_data")
        split_data_table = wandb.Table(columns=["image_path", "image", "label", "split"])
        i = 0
        for (train_image, train_label), train_image_file_path in tf.data.Dataset.zip(
                datasets=(train_ds, train_ds_files)):
            logger.debug(f"Uploading training image [{i + 1}/{num_train_samples}]: {train_image_file_path}")
            split_data_table.add_data(train_image_file_path, wandb.Image(train_image.numpy()), train_label.numpy(),
                                      "train")
            i += 1
        i = 0
        for (val_image, val_label), val_image_file_path in tf.data.Dataset.zip(datasets=(val_ds, val_ds_files)):
            logger.debug(f"Uploading validation image [{i + 1}/{num_val_samples}]: {val_image_file_path}")
            split_data_table.add_data(val_image_file_path, wandb.Image(val_image.numpy()), val_label.numpy(), "val")
            i += 1
        if upload_test_data:
            i = 0
            for (test_image, test_label), test_image_file_path in tf.data.Dataset.zip(
                    datasets=(test_ds, test_ds_files)):
                logger.debug(f"Uploading testing image [{i + 1}/{num_test_samples}]: {test_image_file_path}")
                split_data_table.add_data(test_image_file_path, wandb.Image(test_image.numpy()), "REDACTED", "test")
            i += 1
        hive_data_artifact.add(split_data_table, name=f"{self._wab_group_name}_split_data_{num_samples}")
        upload_split_data_run.log_artifact(hive_data_artifact)
        upload_split_data_run.finish()


if __name__ == '__main__':
    NUM_PHYSICAL_GPUS = 2
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', default='bee', help='Username for the SSH tunnel to the AppMAIS server.')
    parser.add_argument('-p', '--password', help='Password for the SSH tunnel to the AppMAIS server.')
    args = parser.parse_args()
    stft_params = StftParams(
        max_freq=4096,
        min_freq=0,
        window_length=512,
        python_lib=AudioLib.LIBROSA,
        window_type='hann',
        n_fft=2 ** 16,
        num_overlap=2
    )
    piping_detector = PipingDetector(
        overwrite_existing=True,
        train_from_scratch=True,
        use_grayscale=True,
        stft_params=stft_params,
        appmais_user_name=args.username,
        appmais_password=args.password,
        hive_names_for_analysis=['AppMAIS2L', 'AppMAIS7L', 'AppMAIS7R'],
        use_gpu=True,
        batch_size=512,
        seed=42
    )

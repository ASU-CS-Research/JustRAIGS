import copy
from typing import Tuple, Union, List, Optional
import tensorflow as tf
import keras_tuner as kt
from loguru import logger
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.keras.losses import Loss
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.eager import context
from keras.engine import base_layer, data_adapter, training_utils
from keras.utils import version_utils, tf_utils, io_utils
from keras import callbacks as callbacks_module
from wandb.sdk.wandb_run import Run


class KerasTunerWaBHyperModel(kt.HyperModel):
    """
    This is the base class for all KerasTuner models that are compatible with Weights and Biases. This class is designed
    to be subclassed by concrete models. This code was intended to run on the lambda machine compute server with a NFS
    mount to the AppMAIS data storage server.

    See Also:
        - :class:`FromScratchHyperModel`
    """

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]], hyperparameters: Optional[kt.HyperParameters] = None,
            resampled_steps_per_epoch: Optional[int] = None):
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
        super().__init__(name='KerasTunerWaBHyperModel', tunable=True)

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
        model_head = tf.keras.layers.Flatten()(base_model.outputs[0])
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

    @property
    def optimizer(self) -> Optional[Optimizer]:
        return self._optimizer


class FromScratchHyperModel(KerasTunerWaBHyperModel):
    """
    This is an example concrete subclass of a KerasTunerWaBHyperModel. By subclassing this hypermodel will be compatible
    with both KerasTuner and Weights and Biases.
    """

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]],
            hyperparameters: Optional[kt.HyperParameters] = None):
        """

        Args:
            input_image_shape (Tuple[int, int, int])
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

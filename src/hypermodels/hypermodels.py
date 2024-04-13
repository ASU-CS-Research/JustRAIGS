import copy
import traceback
from typing import Tuple, Union, List, Optional, Dict, Any
import tensorflow as tf
# import keras_tuner as kt
from keras.callbacks import History
from loguru import logger
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Optimizer, Adam, RMSprop, SGD
from wandb.integration.keras import WandbCallback
from wandb.sdk.wandb_run import Run
import wandb as wab
import sys
from contextlib import redirect_stdout
from src.callbacks.custom import ConfusionMatrixCallback, TrainValImageCallback, GradCAMCallback
from src.models.models import WaBModel, InceptionV3WaBModel, CVAEWaBModel, EfficientNetB7WaBModel
from src.models.cvae import VariationalAutoEncoder
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from wandb.sdk.wandb_config import Config
from enum import Enum


class ModelType(Enum):
    """
    An enumeration of the different types of models that can be trained by the WaB HyperModel.
    """
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    FEATURE_EXTRACTION = 'feature_extraction'


class WaBHyperModel:
    """
    This is an example Weights and Biases Hypermodel without KerasTuner integration (i.e. WaB will handle hyperparameter
    tuning instead of keras). The hypermodel is the class which is in charge of repeatedly instantiating WaB trials.
    Each trial is a unique set of hyperparameters defined in the sweep configuration. The hypermodel is also responsible
    for training the model specified by the trial, and logging the results to WaB.

    See Also:
        - https://docs.wandb.ai/guides/sweeps/hyperparameter-optimization
        - https://docs.wandb.ai/guides/integrations/keras

    """
    def __init__(
            self, train_ds: Dataset, val_ds: Optional[Dataset], test_ds: Dataset, num_classes: int, training: bool,
            batch_size: int, metrics: List[Metric], wab_config_defaults: Optional[Dict[str, Any]] = None,
            hyper_model_name: Optional[str] = 'WaBHyperModel', num_images_to_visualize: Optional[int] = 6):
        """

        Args:
            train_ds (Dataset): The training dataset. If the training flag is set to ``False`` this is expected to be
              the training + validation datasets combined. Otherwise, this is the normal training dataset.
            val_ds (Optional[Dataset]): The validation dataset. If the training flag is set to ``False`` this is
              expected to be ``None`` (as the training and validation datasets will have been combined). Otherwise, this
              is the normal training set.
            test_ds (Dataset): The testing dataset. This dataset will not be used if the training flag is set to
              ``True``. If the training flag is set to ``False`` then this dataset will be used to evaluate the model.
            num_classes (int): The number of classes in the classification problem, determines if a sigmoid or softmax
              unit is leveraged on the final output layer.
            training (bool): A boolean flag indicating if the model is being trained or evaluated (i.e. inference mode).
            batch_size (int): The batch size for the ?? dataset.
            metrics (List[Metric]): A list of metrics to be used to evaluate the model.
            wab_config_defaults (Optional[Dict[str, Any]]): A dictionary containing the default configuration for the
              Weights and Biases sweep configuration object.
            hyper_model_name (Optional[str]): Defaults to ``'WaBHyperModel'``. The name of the hyper model utilized by
              the :meth:`~src.hypermodels.hypermodels.flatten_hyperparameters` method to subset the hyperparameter
              search space for the WaB Sweeper agent.
            num_images_to_visualize (int): The number of images to utilize in visualizations (both during training and
              inference).

        """
        self._train_ds = train_ds
        logger.debug(f"train_ds.element_spec: {train_ds.element_spec}")
        self._val_ds = val_ds
        if val_ds is not None:
            logger.debug(f"val_ds.element_spec: {val_ds.element_spec}")
        self._test_ds = test_ds
        logger.debug(f"test_ds.element_spec: {test_ds.element_spec}")
        # Note: It is assumed here that train, val, and test images are all the same shape:
        self._image_shape_with_batch_dim = tuple(train_ds.element_spec[0].shape)
        self._image_shape_no_batch_dim = self._image_shape_with_batch_dim[1:]
        self._num_classes = num_classes
        self._training = training
        self._batch_size = batch_size
        self._metrics = metrics
        self._wab_config_defaults = wab_config_defaults
        logger.info(f"wab_sweep_config_defaults: {self._wab_config_defaults}")
        # Set default hyperparameter config values that remain fixed:
        # if self._wab_config_defaults is None:
        #     self._wab_config_defaults = {
        #         'image_size': (112, 112),
        #         'batch_size': 32
        #     }
        # logger.info(f"wab_sweep_config_defaults (post-init): {self._wab_config_defaults}")
        # Group name for the Weights and Biases sweep:
        self._wab_group_name = f"{wab.util.generate_id()}"
        self._hyper_model_name = hyper_model_name
        self._hyper_model_type = ModelType.BINARY_CLASSIFICATION if num_classes == 2 else ModelType.MULTICLASS_CLASSIFICATION
        self._num_images_to_visualize = num_images_to_visualize

    @property
    def hyper_model_type(self):
        return self._hyper_model_type

    @property
    def hyper_model_name(self) -> str:
        return self._hyper_model_name

    @hyper_model_name.setter
    def hyper_model_name(self, hyper_model_name: str):
        self._hyper_model_name = hyper_model_name

    def construct_model_run_trial(self):
        """
        This method is invoked REPEATEDLY by the WaB agent for each trial (unique set of hyperparameters). This method
        must instantiate a new model that utilizes the set of hyperparameters specified by the trial. Then this method
        must train the instantiated model, and log the results to WaB. Recall that each trial is a unique set of
        hyperparameters specified by the WaB sweep configuration.

        See Also:
            - https://docs.wandb.ai/guides/sweeps/hyperparameter-optimization
            - https://docs.wandb.ai/guides/integrations/keras

        """
        # Initialize the namespace/container for this particular trial run with WandB:
        wab_trial_run = wab.init(
            project='JustRAIGS', entity='appmais', config=wab.config, group=self._wab_group_name
        )
        # Workaround for exception logging:
        sys.excepthook = exc_handler
        # Wandb agent will override the defaults with the sweep configuration subset it has selected according to the
        # specified 'method' in the config:
        logger.info(f"wandb.config: {wab.config}")
        # The sweep configuration for this particular trial will then be provided to the Model for hyperparameter
        # parsing to ensure the hyperparameters specified are leveraged by the model (i.e. adherence is delegated):
        model = WaBModel(
            wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape_no_batch_dim,
            batch_size=self._batch_size,
            name='WaBModel'
        )
        # Parse optimizer:
        optimizer_config = wab.config['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # Parse loss function:
        loss_function = wab.config['loss']
        if loss_function == 'binary_crossentropy':
            loss = BinaryCrossentropy(from_logits=False)
        else:
            logger.error(f"Unknown loss function: {loss_function} provided in the hyperparameter section of the sweep "
                         f"configuration.")
            exit(1)
        # Parse target conv layer for visualizations during training:
        visualization = wab.config['visualization']
        visualization_logging_frequency_epochs = visualization['epoch_frequency']
        if 'target_layer' in visualization:
            target_conv_layer_name_to_visualize = visualization['target_layer']
            logger.debug(f"Visualizations will target layer: {target_conv_layer_name_to_visualize}")
        else:
            target_conv_layer_name_to_visualize = None
            logger.debug(f"No target layer specified for visualization during training.")
        # .. todo:: Batch size should be known now? Is that why output of the layer is multiple in model.summary()?
        model.build(input_shape=(self._batch_size, *self._image_shape_no_batch_dim))
        # compile the model:
        model.compile(optimizer=optimizer, loss=loss, metrics=self._metrics)
        # .. todo: Should hparams be part of build or constructor?
        # model = model.build(input_shape=self._image_shape_no_batch_dim, trial_hyperparameters=wab.config)
        # Log the model summary to weights and biases console out:
        wab_trial_run.log({"model_summary": model.summary()})
        # Log the model summary to a text file and upload it as an artifact to weights and biases:
        with open("model_summary.txt", "w") as fp:
            with redirect_stdout(fp):
                model.summary()
        model_summary_artifact = wab.Artifact("model_summary", type='model_summary')
        model_summary_artifact.add_file("model_summary.txt")
        wab_trial_run.log_artifact(model_summary_artifact)
        if self._training:
            # Standard training loop:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run, train_ds=self._train_ds,
                val_ds=self._val_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                target_conv_layer_name_to_visualize=target_conv_layer_name_to_visualize,
                visualization_frequency_epochs=visualization_logging_frequency_epochs
            )
        else:
            # Support for final training run performed after model selection process:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run,
                train_ds=self._train_ds, val_ds=self._test_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                visualization_frequency_epochs=visualization_logging_frequency_epochs
            )
        wab_trial_run.finish()
        tf.keras.backend.clear_session()

    @staticmethod
    def run_trial(
            model: Model, num_classes: int, wab_trial_run: Run, train_ds: Dataset, val_ds: Optional[Dataset],
            test_ds: Dataset, num_epochs: int, num_images_to_visualize: int, visualization_frequency_epochs: int,
            target_conv_layer_name_to_visualize: Optional[str] = None) -> History:
        """
        Runs an individual trial (i.e. a unique set of hyperparameters) for the model as part of an overarching sweep.
        This method is responsible for training (i.e. fitting) the model, and maintaining a :class:`keras.callbacks.History`
        object to upload to WaB.

        .. todo:: Add keras early stopping callbacks.

        Args:
            model (:class:`tf.keras.Model`): The model to be trained with the hyperparameters specified by the trial.
            num_classes (int): The number of classes in the classification problem, this information is needed by
              several of the custom callbacks (such as the :class:`~src.callbacks.custom.ConfusionMatrixCallback`).
            wab_trial_run (:class:`~wandb.sdk.wandb_run.Run`): The WandB Run object for the current trial.
            train_ds (:class:`tf.data.Dataset`): The training dataset to which the provided :class:`tf.keras.Model` will
              be fit to.
            val_ds (:class:`tf.data.Dataset`): The validation dataset to which the provided :class:`tf.keras.Model` will
              be evaluated against during training.
            test_ds (:class:`tf.data.Dataset`): The testing dataset (currently not used, but requested for the sake of
              consistency with other methods).
            num_epochs (int): The number of epochs to train the provided :class:`tf.keras.Model` for during fitting.
            num_images_to_visualize (int): The number of images to plot for visualization purposes during training.
            visualization_frequency_epochs (int): The frequency at which to compute and log visualizations during
              training.
            target_conv_layer_name_to_visualize (Optional[str]): The name of the target convolutional layer to be used
              for visualization purposes. If this value is None then related callbacks (such as GradCAM) will not be
              leveraged.

        Returns:
            :class:`keras.callbacks.History`: The history object containing the training and validation metrics for the
            model during the training process.

        """
        # Fit the model:
        # .. todo:: Early stopping callbacks?
        wab_callback = WandbCallback(
            monitor='val_loss',
            verbose=1,
            save_model=True,
            save_graph=True,
            # generator=val_ds,
            # validation_steps=resampled_val_steps_per_epoch,
            # input_type='image',
            # output_type='label',
            # log_evaluation=True
        )
        # Declare custom Callbacks here:
        confusion_matrix_callback = ConfusionMatrixCallback(
            num_classes=num_classes, wab_trial_run=wab_trial_run, validation_data=val_ds, validation_steps=None,
            epoch_frequency=visualization_frequency_epochs
        )
        # Sample the same images from the validation set to use for callbacks during training:
        val_ds_for_visuals = val_ds.take(num_images_to_visualize).unbatch()
        # train_val_image_callback = TrainValImageCallback(
        #     wab_trial_run=wab_trial_run, train_data=train_ds, val_data=val_ds_for_visuals, num_images=6
        # )
        if target_conv_layer_name_to_visualize is not None:
            grad_cam_callback = GradCAMCallback(
                wab_trial_run=wab_trial_run, num_images=num_images_to_visualize, num_classes=num_classes,
                target_conv_layer_name=target_conv_layer_name_to_visualize,
                validation_data=val_ds_for_visuals, validation_steps=None,
                # validation_batch_size=val_ds_for_visuals.element_spec[1].shape[0],
                validation_batch_size=None,
                log_grad_cam_heatmaps=False, epoch_frequency=visualization_frequency_epochs
            )
            callbacks = [wab_callback, confusion_matrix_callback, grad_cam_callback]
        else:
            callbacks = [wab_callback, confusion_matrix_callback]
        if num_classes > 2:
            # .. todo:: Remove once grad_cam_callback works in the multiclass case.
            callbacks = [wab_callback, confusion_matrix_callback]
            logger.error(f"Multiclass classification callbacks not yet implemented for GradCAM. Overriding callbacks to "
                         f"WaBCallback and ConfusionMatrix only!")

        logger.info(f"Callbacks: {callbacks}")
        # Fit the model and log the trial results to WaB:
        try:
            trial_history = model.fit(
                train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks
            )
        except Exception as err:
            logger.exception(f"Exception occurred during trial run: {err}")
            traceback.print_exc()
            exit(1)
        return trial_history


class EfficientNetB7WaBHyperModel(WaBHyperModel):
    """
    An example of a WaB Hypermodel that leverages the :class:`~src.models.models.EfficientNetB7WaBModel`. This class is
    responsible for constructing unique instances of the :class:`~src.models.models.EfficientNetB7WaBModel` for each trial
    (unique set of hyperparameters) and training the model.

    .. todo:: Remove duplicate code by refactoring the base class. The only appreciable difference between this class
         and the base WaBHyperModel class is the type of model being instantiated.

    """

    def __init__(self, train_ds: Dataset, val_ds: Optional[Dataset], test_ds: Dataset, num_classes: int, training: bool,
                 batch_size: int, metrics: List[Metric], num_images_to_visualize: Optional[int] = 6):
        super().__init__(train_ds, val_ds, test_ds, num_classes, training, batch_size, metrics,
                         hyper_model_name="EfficientNetB7", num_images_to_visualize=num_images_to_visualize)

    def construct_model_run_trial(self):
        """
        This method is invoked REPEATEDLY by the WaB agent for each trial (unique set of hyperparameters). This method
        must instantiate a new model that utilizes the set of hyperparameters specified by the trial. Then this method
        must train the instantiated model, and log the results to WaB. Recall that each trial is a unique set of
        hyperparameters specified by the WaB sweep configuration.

        See Also:
            - https://docs.wandb.ai/guides/sweeps/hyperparameter-optimization
            - https://docs.wandb.ai/guides/integrations/keras

        """
        # Initialize the namespace/container for this particular trial run with WandB:
        wab_trial_run = wab.init(
            project='JustRAIGS', entity='appmais', config=wab.config, group=self._wab_group_name
        )
        # Workaround for exception logging:
        sys.excepthook = exc_handler
        # Wandb agent will override the defaults with the sweep configuration subset it has selected according to the
        # specified 'method' in the config:
        logger.info(f"wandb.config: {wab.config}")
        # The sweep configuration for this particular trial will then be provided to the Model for hyperparameter
        # parsing to ensure the hyperparameters specified are leveraged by the model (i.e. adherence is delegated):
        model = EfficientNetB7WaBModel(
            wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape_no_batch_dim,
            batch_size=self._batch_size, num_classes=self._num_classes, name='EfficientNetB7WaBModel'
        )
        # Parse optimizer:
        optimizer_config = wab.config['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=optimizer_learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=optimizer_learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=optimizer_learning_rate)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # Parse loss function:
        loss_function = wab.config['loss']
        if loss_function == 'binary_crossentropy':
            loss = BinaryCrossentropy(from_logits=False)
        elif loss_function == 'categorical_crossentropy':
            loss = CategoricalCrossentropy(from_logits=False)
        else:
            logger.error(f"Unknown loss function: {loss_function} provided in the hyperparameter section of the sweep "
                         f"configuration.")
            exit(1)
        visualizations = wab.config['visualization']
        visualization_frequency_epochs = visualizations['epoch_frequency']
        target_conv_layer_to_visualize = visualizations['target_layer']
        logger.info(f"Logging visualizations at a frequency (epochs) of: {visualization_frequency_epochs}")
        # .. todo:: Batch size should be known now? Is that why output of the layer is multiple in model.summary()?
        model.build(input_shape=(self._batch_size, *self._image_shape_no_batch_dim))
        # compile the model:
        model.compile(optimizer=optimizer, loss=loss, metrics=self._metrics)
        # .. todo: Should hparams be part of build or constructor?
        # model = model.build(input_shape=self._image_shape_no_batch_dim, trial_hyperparameters=wab.config)
        # Log the model summary to weights and biases console out:
        wab_trial_run.log({"model_summary": model.summary()})
        # Log the model summary to a text file and upload it as an artifact to weights and biases:
        with open("model_summary.txt", "w") as fp:
            with redirect_stdout(fp):
                model.summary()
        model_summary_artifact = wab.Artifact("model_summary", type='model_summary')
        model_summary_artifact.add_file("model_summary.txt")
        wab_trial_run.log_artifact(model_summary_artifact)
        if self._training:
            # Standard training loop:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run, train_ds=self._train_ds,
                val_ds=self._val_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                visualization_frequency_epochs=visualization_frequency_epochs,
                target_conv_layer_name_to_visualize=target_conv_layer_to_visualize
            )
        else:
            # Support for final training run performed after model selection process:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run,
                train_ds=self._train_ds, val_ds=self._test_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                visualization_frequency_epochs=visualization_frequency_epochs,
                target_conv_layer_name_to_visualize=target_conv_layer_to_visualize
            )
        wab_trial_run.finish()
        tf.keras.backend.clear_session()


class InceptionV3WaBHyperModel(WaBHyperModel):
    """
    An example of a WaB Hypermodel that leverages the :class:`~src.models.models.InceptionV3WaBModel`. This class is
    responsible for constructing unique instances of the :class:`~src.models.models.InceptionV3WaBModel` for each trial
    (unique set of hyperparameters) and training the model.

    .. todo:: Remove duplicate code by refactoring the base class. The only appreciable difference between this class
         and the base WaBHyperModel class is the type of model being instantiated.

    """

    def __init__(self, train_ds: Dataset, val_ds: Optional[Dataset], test_ds: Dataset, num_classes: int, training: bool,
                 batch_size: int, metrics: List[Metric], hyper_model_name: Optional[str] = "InceptionV3",
                 num_images_to_visualize: Optional[int] = 6):
        super().__init__(
            train_ds, val_ds, test_ds, num_classes, training, batch_size, metrics, hyper_model_name=hyper_model_name,
            num_images_to_visualize=num_images_to_visualize
        )
        self._hyper_model_type = ModelType.BINARY_CLASSIFICATION if num_classes == 2 else ModelType.MULTICLASS_CLASSIFICATION

    def construct_model_run_trial(self):
        """
        This method is invoked REPEATEDLY by the WaB agent for each trial (unique set of hyperparameters). This method
        must instantiate a new model that utilizes the set of hyperparameters specified by the trial. Then this method
        must train the instantiated model, and log the results to WaB. Recall that each trial is a unique set of
        hyperparameters specified by the WaB sweep configuration.

        See Also:
            - https://docs.wandb.ai/guides/sweeps/hyperparameter-optimization
            - https://docs.wandb.ai/guides/integrations/keras

        """
        # Initialize the namespace/container for this particular trial run with WandB:
        wab_trial_run = wab.init(
            project='JustRAIGS', entity='appmais', config=wab.config, group=self._wab_group_name
        )
        # Workaround for exception logging:
        sys.excepthook = exc_handler
        # Wandb agent will override the defaults with the sweep configuration subset it has selected according to the
        # specified 'method' in the config:
        logger.info(f"wandb.config: {wab.config}")
        # The sweep configuration for this particular trial will then be provided to the Model for hyperparameter
        # parsing to ensure the hyperparameters specified are leveraged by the model (i.e. adherence is delegated):
        model = InceptionV3WaBModel(
            wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape_no_batch_dim,
            batch_size=self._batch_size, num_classes=self._num_classes, name='InceptionV3WaBModel'
        )
        # Parse optimizer:
        optimizer_config = wab.config['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # Parse loss function:
        loss_function = wab.config['loss']
        if loss_function == 'binary_crossentropy':
            loss = BinaryCrossentropy(from_logits=False)
        else:
            logger.error(f"Unknown loss function: {loss_function} provided in the hyperparameter section of the sweep "
                         f"configuration.")
            exit(1)
        visualizations = wab.config['visualization']
        visualization_frequency_epochs = visualizations['epoch_frequency']
        target_conv_layer_to_visualize = visualizations['target_layer']
        # .. todo:: Batch size should be known now? Is that why output of the layer is multiple in model.summary()?
        model.build(input_shape=(self._batch_size, *self._image_shape_no_batch_dim))
        # compile the model:
        model.compile(optimizer=optimizer, loss=loss, metrics=self._metrics)
        # .. todo: Should hparams be part of build or constructor?
        # model = model.build(input_shape=self._image_shape_no_batch_dim, trial_hyperparameters=wab.config)
        # Log the model summary to weights and biases console out:
        wab_trial_run.log({"model_summary": model.summary()})
        # Log the model summary to a text file and upload it as an artifact to weights and biases:
        with open("model_summary.txt", "w") as fp:
            with redirect_stdout(fp):
                model.summary()
        model_summary_artifact = wab.Artifact("model_summary", type='model_summary')
        model_summary_artifact.add_file("model_summary.txt")
        wab_trial_run.log_artifact(model_summary_artifact)
        if self._training:
            # Standard training loop:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run, train_ds=self._train_ds,
                val_ds=self._val_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                visualization_frequency_epochs=visualization_frequency_epochs,
                target_conv_layer_name_to_visualize=target_conv_layer_to_visualize
            )
        else:
            # Support for final training run performed after model selection process:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run,
                train_ds=self._train_ds, val_ds=self._test_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize,
                visualization_frequency_epochs=visualization_frequency_epochs,
                target_conv_layer_name_to_visualize=target_conv_layer_to_visualize
            )
        wab_trial_run.finish()
        tf.keras.backend.clear_session()


class CVAEFeatureExtractorHyperModel(WaBHyperModel):
    """
    An example of a WaB Hypermodel that leverages the :class:`~src.models.models.CVAEFeatureExtractorWaBModel`. This
    class is responsible for constructing unique instances of the :class:`~src.models.models.CVAEFeatureExtractorWaBModel`
    for each trial (unique set of hyperparameters) and training the model that performs feature extraction.
    """
    def __init__(self, train_ds: Dataset, val_ds: Optional[Dataset], test_ds: Dataset, num_classes: int, training: bool,
                 batch_size: int, metrics: List[Metric], hyper_model_name: Optional[str] = "CVAEFeatureExtractorHyperModel",
                 num_images_to_visualize: Optional[int] = 6):
        super().__init__(
            train_ds, val_ds, test_ds, num_classes, training, batch_size, metrics, hyper_model_name=hyper_model_name,
            num_images_to_visualize=num_images_to_visualize
        )

    def construct_model_run_trial(self):
        """
        This method is invoked REPEATEDLY by the WaB agent for each trial (unique set of hyperparameters). This method
        must instantiate a new model that utilizes the set of hyperparameters specified by the trial. Then this method
        must train the instantiated model, and log the results to WaB. Recall that each trial is a unique set of
        hyperparameters specified by the WaB sweep configuration.

        .. todo:: Remove duplicate code by refactoring the base class. The only appreciable difference between this
             class and the base WaBHyperModel class is the type of model instantiated.

        See Also:
            - https://docs.wandb.ai/guides/sweeps/hyperparameter-optimization
            - https://docs.wandb.ai/guides/integrations/keras

        """
        # Initialize the namespace/container for this particular trial run with WandB:
        wab_trial_run = wab.init(
            project='JustRAIGS', entity='appmais', config=wab.config, group=self._wab_group_name,
            tags=['feature_extraction']
        )
        # Workaround for exception logging:
        sys.excepthook = exc_handler
        # Wandb agent will override the defaults with the sweep configuration subset it has selected according to the
        # specified 'method' in the config:
        # model = CVAEWaBModel(
        #     wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape_no_batch_dim,
        #     batch_size=self._batch_size, name='CVAEFeatureExtractorWaBModel', num_classes=self._num_classes
        # )
        model = VariationalAutoEncoder(
            wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, batch_size=self._batch_size,
            input_shape=self._image_shape_no_batch_dim, num_classes=self._num_classes, original_dim=64,
            name='VariationalAutoEncoder'
        )
        # Ensure required parameters are present in the sweep configuration:
        if 'feature_extraction' not in wab.config:
            logger.error(f"Feature extraction hyperparameters not found in the sweep configuration.")
            exit(1)
        else:
            feature_extraction = wab.config['feature_extraction']
        # Parse required parameters:
        latent_dim = feature_extraction['latent_dim']
        optimizer_params = feature_extraction['optimizer']
        optimizer_type = optimizer_params['type']
        optimizer_learning_rate = optimizer_params['learning_rate']
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        loss = tf.keras.losses.MeanSquaredError()
        # Build the model:
        # model.build(input_shape=self._image_shape_with_batch_dim)
        # Compile the model:
        # model.compile(optimizer=optimizer, loss=loss, metrics=self._metrics)
        # Log the model summary to weights and biases console out:
        # wab_trial_run.log({"model_summary": model.summary()})
        # # Log the model summary to a text file and upload it as an artifact to weights and biases:
        # with open("model_summary.txt", "w") as fp:
        #     with redirect_stdout(fp):
        #         model.summary()
        # model_summary_artifact = wab.Artifact("model_summary", type='model_summary')
        # model_summary_artifact.add_file("model_summary.txt")
        # wab_trial_run.log_artifact(model_summary_artifact)
        if self._training:
            # Standard training loop:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run, train_ds=self._train_ds,
                val_ds=self._val_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize
            )
        else:
            # Support for final training run performed after model selection process:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run,
                train_ds=self._train_ds, val_ds=self._test_ds, test_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'], num_images_to_visualize=self._num_images_to_visualize
            )
        # .. todo:: Do we want a separate trial run for the feature extractor? If so, don't call finish here:
        wab_trial_run.finish()
        tf.keras.backend.clear_session()


def exc_handler(exc_type, exc, tb):
    """
    This is a workaround for WaB not logging exceptions properly. This function is intended to be used as the exception
    handler. This method may not be necessary in the future if WaB fixes its stack trace logging.
    """
    logger.exception(f"EXCEPTION")
    print("EXCEPTION")
    traceback.print_exception(exc_type, exc, tb)


def flatten_hyperparameters(hyperparameters: Config, model_name: str, model_type: ModelType) -> Dict[str, Any]:
    """
    This method is responsible for flattening the hyperparameters specified in the sweep configuration into a single
    dictionary, for convenience's sake. This method expects a wandb_config dictionary laid out like so::
    {
        'global': {
            'parameters': {
                'binary_classification': {
                    parameters: {
                        *global binary classification hyperparameters*
                    }
                },
                ...
            }
        },
        'binary_classification': {
            parameters: {
                'InceptionV3': {
                    parameters: {
                        *InceptionV3 specific hyperparameters*
                    }
                },
                'EfficientNetB0': {
                    parameters: {
                        *EfficientNetB0 specific hyperparameters*
                    }
                },
                ...
            }
        }
    }

    Args:
        hyperparameters (Dict[str, Any]): Configuration
        model_name (str):
        model_type (ModelType):

    Returns:
        Dict[str, Any]: The flattened hyperparameters dictionary. The example config above with the parameters
          model_name='InceptionV3' and model_type=ModelType.BINARY_CLASSIFICATION would be flattened to:
          {
              *global binary classification hyperparameters*,
              *InceptionV3 specific hyperparameters*
          }
          With any overlaps between the two dictionaries resolved by the InceptionV3 specific hyperparameters.
    """
    # Initialize the flattened hyperparameters dictionary:
    flattened_hyperparameters = {}
    # Extract the global hyperparameters:
    global_hyperparameters = hyperparameters['global']['parameters'][model_type.value]
    # Extract the model specific hyperparameters:
    try:
        model_specific_hyperparameters = hyperparameters[model_type.value]['parameters'][model_name]
    except KeyError as err:
        logger.error(f"Missing model \'{model_name}\' from the \'{model_type.value}\' section of \'hparams.json\'")
        exit(1)
    # Flatten the hyperparameters:
    flattened_hyperparameters.update(global_hyperparameters['parameters'])
    flattened_hyperparameters.update(model_specific_hyperparameters['parameters'])
    return flattened_hyperparameters

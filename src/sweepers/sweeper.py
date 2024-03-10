import enum
import os
import socket
import sys
import traceback
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any, Dict, Union, List, Tuple, Optional
import numpy as np
from loguru import logger
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from wandb.apis.public import Run
from wandb.integration.keras import WandbCallback
import wandb as wab
import wandb.util
from wandb.sdk.wandb_config import Config
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tensorflow.keras.callbacks import Callback
from src.metrics.metrics import BalancedBinaryAccuracy

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


class DatasetSplit(enum.Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    TRAIN_AND_VALIDATION = 4


# @tf.keras.utils.register_keras_serializable(name='ConfusionMatrixCallback')
class ConfusionMatrixCallback(Callback):
    """
    Plots the confusion matrix on the validation dataset after training has finished and uploads the results to WandB.
    """

    def __init__(self, num_classes: int, wab_trial_run: Run, validation_data: Dataset, validation_steps: Optional[int] = None):
        """

        Args:
            num_classes (int): The number of classes for the classification problem.
            wab_trial_run (Run): An instance of the WandB Run class for the current trial.
            validation_data (Dataset): A tensorflow Dataset object containing the validation data. This dataset may or
              may not have infinite cardinality at runtime (as a result of oversampling). The dataset will yield (x, y)
              tuples of validation data.
            validation_steps (Optional[int]): The number of steps/batches to be considered one epoch for the validation
              dataset. This value must be provided if the validation dataset has infinite cardinality at runtime.

        """
        self._num_classes = num_classes
        self._wab_trial_run = wab_trial_run
        self._validation_data = validation_data
        self._validation_steps = validation_steps

    def on_train_end(self, logs=None):
        logger.debug(f"Generating confusion matrix for {self._num_classes} classes on the validation dataset...")
        # Check to see if the provided validation dataset is infinite:
        if self._validation_data.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
            if self._validation_steps is None:
                raise ValueError("Since the provided validation dataset is infinite, the number of validation steps "
                                 "must be specified.")
        y_pred = np.array([])
        y_true = np.array([])
        class_names = np.array([])
        for i, batch in enumerate(self._validation_data):
            if self._validation_steps is not None:
                if i == self._validation_steps:
                    break
                images = batch[0]
                labels = batch[1]
                y_true = np.concatenate((y_true, labels.numpy()))
                y_pred_batch = self.model.predict_on_batch(images).squeeze()
                class_names_batch = np.array(['Piping' if label.numpy() == 1 else 'NoPiping' for label in labels])
                y_pred = np.concatenate((y_pred, y_pred_batch))
                class_names = np.concatenate((class_names, class_names_batch))
        y_pred = np.array([0 if pred <= 0.5 else 1 for pred in y_pred])
        y_true = y_true.squeeze()
        confusion_matrix = tf.math.confusion_matrix(
            labels=y_true, predictions=y_pred, num_classes=self._num_classes, dtype=tf.int32
        )
        ax = sns.heatmap(confusion_matrix, annot=True, fmt="g")
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.set(font_scale=1.4)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        ax.xaxis.set_ticklabels(np.unique(class_names))
        ax.yaxis.set_ticklabels(np.unique(class_names))
        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        self._wab_trial_run.log({'confusion_matrix': wab.Image(fig)})
        plt.clf()
        plt.close(fig)


# @tf.keras.utils.register_keras_serializable(name='GradCAMCallback')
class GradCAMCallback(Callback):
    """
    Runs Grad-CAM on a subset of random images from the validation set (where the predicted class label is correct) and
    uploads the results to WandB. This callback is designed to execute only when training is finished.
    """

    def __init__(
            self, num_images: int, num_classes: int, wab_trial_run: Run, target_conv_layer_name: str, validation_data: Dataset,
            validation_batch_size: int, validation_steps: Optional[int] = None,
            log_conv2d_output: Optional[bool] = False, log_grad_cam_heatmaps: Optional[bool] = False,
            log_target_conv_layer_kernels: Optional[bool] = False, grad_cam_heatmap_alpha_value: Optional[float] = 0.4):
        """

        Args:
            num_classes (int): The number of classes for the classification problem.
            num_images (int): The number of validation set images to run Grad-CAM on (where the class label is
              predicted correctly).
            num_classes (int): The number of classes for the classification problem.
            wab_trial_run (Run): An instance of the WandB Run class for the current trial.
            target_conv_layer_name (str): The name of the convolutional layer that Grad-CAM should utilize for
              visualizations.
            validation_data (Dataset): A tensorflow Dataset object containing the validation data. This dataset may or
              may not have infinite cardinality at runtime (as a result of oversampling). The dataset will yield (x, y)
              tuples of validation data.
            validation_batch_size (int): The number of images in a validation dataset batch. In its current state this
              callback will un-batch the provided dataset. This argument is required so that it is possible to re-batch
              the dataset after Grad-CAM images are generated.
            validation_steps (Optional[int]): The number of steps/batches to be considered one epoch for the validation
              dataset. This value must be provided if the validation dataset has infinite cardinality at runtime.
            log_conv2d_output (Optional[bool]): A boolean flag indicating whether the raw output of the Convolutional
              layer specified by :attr:`_target_conv_layer_name` should be plotted and logged (sent northbound) to
              WandB.
            log_grad_cam_heatmaps (Optional[bool]): A boolean flag indicating whether the raw Grad-CAM heatmaps should
              be plotted and logged (sent northbound) to WandB.
            log_target_conv_layer_kernels (Optional[bool]): A boolean flag indicating whether the learned kernels of the
              Convolutional layer specified by :attr:`_target_conv_layer_name` should be plotted and logged (sent
              northbound) to WandB.
            grad_cam_heatmap_alpha_value (Optional[float]): The alpha value to use when superimposing the Grad-CAM
              heatmaps onto the source image. This value should be between 0 and 1, and defaults to ``0.4``.

        """
        self._num_images = num_images
        self._num_classes = num_classes
        self._wab_trial_run = wab_trial_run
        self._target_conv_layer_name = target_conv_layer_name
        self._validation_data = validation_data
        self._validation_batch_size = validation_batch_size
        self._validation_steps = validation_steps
        self._log_conv2d_output = log_conv2d_output
        self._log_grad_cam_heatmaps = log_grad_cam_heatmaps
        self._log_target_conv_layer_kernels = log_target_conv_layer_kernels
        self._grad_cam_heatmap_alpha_value = grad_cam_heatmap_alpha_value

    def on_train_end(self, logs=None):
        logger.debug(f"Generating Grad-CAM visuals for {self._num_images} on the validation dataset...")
        # Get image shape from the dataset (excluding the batch dimension):
        image_shape = tuple(self._validation_data.element_spec[0].shape[1:])
        # Check to see if the provided validation dataset is infinite:
        if self._validation_data.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
            if self._validation_steps is None:
                raise ValueError("Since the provided validation dataset is infinite, the number of validation steps "
                                 "must be specified.")
        self._validation_data = self._validation_data.unbatch()
        num_correct_predictions = 0
        correctly_predicted_random_images = np.zeros((self._num_images, *image_shape))
        correctly_predicted_random_labels = np.zeros((self._num_images, 1))
        for i, (image, label) in enumerate(self._validation_data):
            # logger.debug(f"num_correct_predictions: {num_correct_predictions}")
            # logger.debug(f"self._num_images: {self._num_images}")
            # logger.debug(f"i: {i}")
            if num_correct_predictions == self._num_images:
                break
            # Model expects input to be (None, width, height, num_channels) so prepend batch dimension:
            image = tf.expand_dims(image, axis=0)
            y_pred_prob = tf.squeeze(self.model(image, training=False), axis=-1)
            y_pred = 0 if y_pred_prob <= 0.5 else 1
            if y_pred == label.numpy():
                correctly_predicted_random_images[num_correct_predictions] = image.numpy()
                correctly_predicted_random_labels[num_correct_predictions] = label.numpy()
                num_correct_predictions += 1
        # Compute (and optionally log) the Grad-CAM heatmaps for the random validation set images:
        heatmaps = self._get_grad_cam_activation_heatmap_for_images(
            images=correctly_predicted_random_images, num_classes=self._num_classes,
            log_heatmap=self._log_grad_cam_heatmaps, log_conv_output=self._log_grad_cam_heatmaps,
            log_kernels=self._log_target_conv_layer_kernels
        )
        # Plot the Grad-CAM heatmaps on-top of the original images (e.g. the proper Grad-CAM visualization):
        for i, image in enumerate(correctly_predicted_random_images):
            heatmap = heatmaps[i]
            # Rescale heatmap to range 0 - 255:
            heatmap = np.uint8(255 * heatmap)
            jet_cm = cm.get_cmap("jet")
            # Replace the alpha channel with the constant heatmap alpha value:
            jet_colors = jet_cm(np.arange(256))[:, :3]
            # jet_heatmap = jet_colors[heatmap]
            # jet_colors = jet_cm(np.arange(jet_cm.N))[...]
            # jet_colors[:, -1] = np.full((jet_cm.N,), heatmap_alpha)
            # custom_jet_cm = matplotlib.colors.ListedColormap(jet_colors)
            jet_heatmap = jet_colors[heatmap]
            # Create image with RGB colorized heatmap:
            jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
            jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
            # Superimpose the heatmap on original image:
            superimposed_img = jet_heatmap * self._grad_cam_heatmap_alpha_value + (image * 255)
            superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
            fig = plt.figure(num=1)
            plt.imshow(superimposed_img, cmap=jet_cm)
            plt.suptitle(f"Grad-CAM at Layer {self._target_conv_layer_name}")
            plt.colorbar()
            plt.grid(None)
            # plt.show()
            self._wab_trial_run.log({f'grad_cam_{self._target_conv_layer_name}': wab.Image(fig)})
            plt.clf()
            plt.close(fig)
        # Re-batch the validation dataset:
        self._validation_data = self._validation_data.batch(batch_size=self._validation_batch_size)

    def _get_grad_cam_activation_heatmap_for_images(
            self, images: np.ndarray, num_classes: int, log_heatmap: Optional[bool] = False, log_conv_output: Optional[bool] = False,
            log_kernels: Optional[bool] = False) -> np.ndarray:
        """
        Computes the Grad-CAM heatmaps for the provided images. The heatmaps are the results of Grad-CAM prior to
        superposition on-top of the source image. This method will additionally optionally log the computed heatmaps to
        WandB, as well as the raw output of the target convolutional layer specified by :attr:`_target_conv_layer_name`.

        See Also:
            - https://keras.io/examples/vision/grad_cam/
            - Deep Learning with Python (book) by Francois Chollet

        Args:
            images (np.ndarray): A numpy array containing the images for which Grad-CAM heatmaps should be computed on.
              Images are expected to be of shape: ``(width, height, num_channels)`` which excludes the batch dimension.
            num_classes (int): The number of classes for the classification problem. Utilized by Grad-CAM to extract the
              predicted probabilities for the target image.
            log_heatmap (Optional[bool]): A boolean flag indicating whether the raw Grad-CAM heatmaps should be plotted
              and logged (sent northbound) to WandB.
            log_conv_output (Optional[bool]): A boolean flag indicating whether the raw output of the Convolutional
              layer specified by :attr:`_target_conv_layer_name` should be plotted and logged (sent northbound) to
              WandB.
            log_kernels (Optional[bool]): A boolean flag indicating whether the learned kernels of the Convolutional
              layer specified by :attr:`_target_conv_layer_name` should be plotted and logged (sent northbound) to
              WandB.

        Returns:

        """
        if log_kernels:
            # Visualize the kernels for the target convolutional layer and upload to WandB.
            last_conv_layer = self.model.get_layer(self._target_conv_layer_name)
            num_kernels = last_conv_layer.kernel.shape[-2]
            fig = plt.figure(num=1)
            for i in range(num_kernels):
                kern = last_conv_layer.kernel.numpy()[..., i, :]
                plt.title(f"Kernel [{i + 1}/{num_kernels}]")
                plt.imshow(kern, cmap='gray')
                plt.colorbar()
                plt.grid(None)
                self._wab_trial_run.log({f'last_conv_layer_kernels': wab.Image(fig)})
                # plt.show()
                plt.clf()
        target_conv_layer = self.model.get_layer(self._target_conv_layer_name)
        conv_layer_output_activation_shape = target_conv_layer.output_shape[1:-1]
        num_images = images.shape[0]
        heatmaps = np.zeros((num_images, *conv_layer_output_activation_shape))
        # New model to map the input image to the activations of the last conv layer as well as the output predictions:
        grad_cam_model = tf.keras.Model(
            self.model.inputs, [target_conv_layer.output, self.model.output], name='grad_cam_model'
        )
        pred_index = None
        for i, image in enumerate(images):
            image = np.expand_dims(image, axis=0)
            # Compute the gradient of the top predicted class for the input image with respect to the activations of the
            # last conv layer:
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_cam_model(image, training=False)
                # preds is now shape (num_classes, 1) and last_conv_layer_output is now shape (1, 112, 112, 1)
                if pred_index is None:
                    # The class index associated with the highest predicted probability:
                    pred_index = tf.argmax(preds[0])
                if num_classes > 2:
                    # The predicted probability of the highest probability class:
                    class_channel = preds[:, pred_index]
                else:
                    # The predicted probability of the highest probability class. Since we are doing binary
                    # classification there is only one class to index into:
                    class_channel = preds[0][0]
            # The gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the
            # last conv layer:
            grads = tape.gradient(class_channel, last_conv_layer_output)
            # Create a vector where each entry is the mean intensity of the gradient over a specific feature map
            # channel:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            num_channels = last_conv_layer_output.shape[-1]
            num_rows = 2
            num_cols = 4
            ''' Convolutional layer output for each kernel: '''
            if log_conv_output:
                fig = plt.figure(num=1)
                if num_channels == 1:
                    conv_output = last_conv_layer_output[0, ..., 0]
                    plt.suptitle(f"{target_conv_layer.name} Output for Kernel")
                    plt.imshow(conv_output, cmap='viridis', aspect='auto')
                else:
                    plt.suptitle(f"{target_conv_layer.name} Output for Each Kernel")
                    # plt.suptitle(f"Conv Layer Output\nEvent: {event_type} on {date_str} at {time_str}")
                    for j in range(num_channels):
                        conv_output = last_conv_layer_output[0, ..., j]
                        ax = plt.subplot(num_rows, num_cols, j + 1)
                        ax.set_xticks(np.arange(0, image.shape[1], 25.0))
                        ax.set_yticks(np.arange(0, image.shape[1], 25.0))
                        ax.imshow(conv_output, cmap='viridis', aspect='auto')
                        ax.title.set_text(f"k{j}")
                plt.tight_layout()
                # plt.show()
                self._wab_trial_run.log({f'{target_conv_layer.name}_output': wab.Image(fig)})
            # Multiply each channel in the feature map array by "how important this channel is" with regard to the top
            # predicted class. Then sum all channels to obtain the heatmap class activation:
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            # Normalize the heatmap between 0 and 1 for visualization purposes:
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            # Plot the image for debugging purposes:
            if log_heatmap:
                plt.matshow(heatmap.numpy(), cmap='jet')
                plt.title(f"{target_conv_layer.name} Grad-CAM Heatmap")
                plt.colorbar()
                plt.grid(None)
                # plt.show()
                fig = plt.gcf()
                # Send northbound to WandB:
                self._wab_trial_run.log({f'weight_heatmap_{target_conv_layer.name}': wab.Image(fig)})
                plt.clf()
                plt.close(fig)
            heatmaps[i] = heatmap.numpy()
        return heatmaps


@tf.keras.utils.register_keras_serializable(name='PipingDetectorWabModel')
class PipingDetectorWabModel(Sequential):

    def __init__(self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, input_shape: Tuple[int, int, int], *args, **kwargs):
        """

        Args:
            wab_trial_run (Optional[Run]): The WandB Run object for the current trial. Used to log output to the same
              namespaced location in WandB. Note that this parameter is optional in the event that the model is being
              loaded from a saved model format (e.g. h5) in which case the user may not wish to log metrics to the same
              trial as the one that generated the saved model. During training it is expected that this value is not
              None.
            trial_hyperparameters (`Config`): The hyperparameters for this particular trial. These are provided by
              the WaB sweep agent as a subset of the total hyperparameter space.
            input_shape:
            *args:
            **kwargs:
        """
        self._wab_trial_run = wab_trial_run
        logger.debug(f"Initializing via call to super()...")
        super().__init__(*args, **kwargs)
        logger.debug(f"Initializing locally...")
        self._trial_hyperparameters = trial_hyperparameters
        self._input_shape = input_shape
        # Build the model with the hyperparameters for this particular trial.
        # The concrete subclass should know which hyperparameters are pertinent to it for the particular trial.
        kernel_size = trial_hyperparameters['kernel_size']
        num_nodes_conv2d_1 = trial_hyperparameters['num_nodes_conv_1']
        if 'num_nodes_conv_2' in trial_hyperparameters:
            num_nodes_conv2d_2 = trial_hyperparameters['num_nodes_conv_2']
        else:
            num_nodes_conv2d_2 = None
        conv_layer_activation_function = trial_hyperparameters['conv_layer_activation_function']
        # try:
        #     num_nodes_conv2d_2 = trial_hyperparameters['num_nodes_conv2d_2']
        # except KeyError:
        #     pass
        # Construct the layers to append to this model (i.e. self):
        self._input_layer = tf.keras.layers.Input(shape=input_shape, name='input_layer')
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
            pool_size=(1, input_shape[0]),
            strides=(1, input_shape[1]),
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
        # logger.debug(f"self._repo_root_dir: {self._repo_root_dir}")
        # self._saved_model_dir = os.path.abspath(os.path.join(self._repo_root_dir, 'data/saved_models'))
        # # logger.debug(f"self._saved_model_dir: {self._saved_model_dir}")
        # if not os.path.exists(self._saved_model_dir):
        #     # logger.debug(f"Creating output directory: {self._saved_model_dir}")
        #     os.makedirs(self._saved_model_dir)
        # self._saved_model_path = os.path.abspath(os.path.join(self._saved_model_dir, 'piping_detector.keras'))
        # logger.warning(f"self._saved_model_path (on init): {self._saved_model_path}")

    def get_config(self):
        base_config = super().get_config()
        # .. todo:: Should the wab_trial_run object be serialized in the config?
        config = {
            'wab_trial_run': None, 'trial_hyperparameters': self._trial_hyperparameters.as_dict(),
            'input_shape': self._input_shape
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
            PipeDetectorWabModel: The model constructed from the provided configuration dictionary. Weights will be
            restored

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
            args[0], custom_objects={"PipingDetectorWabModel": PipingDetectorWabModel}
        )
        # loaded_model.compile(optimizer=self._trial_hyperparameters['optimizer'], loss='binary_crossentropy')
        np.testing.assert_equal(self.get_weights(), loaded_model.get_weights()), f"Saved model weight assertion failed. Weights were most likley saved incorrectly"


    # def build(self, input_shape: Tuple[int, int, int]) -> Model:
    #     # Append the constructed layers:
    #     self.layers.append(self._input_layer)
    #     self.layers.append(self._conv_2d_1)
    #     self.layers.append(self._average_pool_2d_1)
    #     self.layers.append(self._flatten_1)
    #     self.layers.append(self._output_layer)

    # def call(self, inputs, *args, **kwargs):
    #     # Forward pass:
    #     return self(inputs=inputs, *args, **kwargs)

    # def on_epoch_end(self, epoch, logs=None):
    #     """
    #     .. todo:: Docstrings.
    #
    #     See Also:
    #         https://www.tensorflow.org/guide/keras/writing_your_own_callbacks#global_methods
    #
    #     Args:
    #         epoch:
    #         logs:
    #
    #     Returns:
    #
    #     """
    #     # .. todo:: Wandb logging. Or should that be handled by the WandbCallback?
    #     raise NotImplementedError()

    # def train_step(self, data) -> Dict[str, Any]:
    #     """
    #     Overrides the :meth:`Model.train_step` method of the base class to customize what happens during calls to
    #     :meth:`Model.fit` during training.
    #
    #     Args:
    #         data (Any): The data to train on. This could be a tuple of numpy arrays ``(x, y)`` if :meth:`Model.fit` was
    #           called with an ``x`` and ``y`` pair. If :meth:`Model.fit` was called with a TensorFlow dataset as in:
    #           ``model.fit(dataset, ...)`` then ``data`` will be what is yielded by the dataset iterator at each batch.
    #
    #     See Also:
    #         - https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    #         - https://docs.wandb.ai/tutorials/tensorflow_sweeps#%EF%B8%8F-build-a-simple-classifier-mlp
    #         - https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#a_first_simple_example
    #
    #     """
    #     # Unpack the data (the structure depends on the model and what is passed to `fit()`).
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         # Forward pass:
    #         y_pred = self(x, training=True)
    #         # Compute the loss value (the loss function is configured in `compile()`):
    #         loss = self.compute_loss(y=y, y_pred=y_pred)
    #     # Compute the gradients:
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     # Update the weights of the model via the optimizer (take a step along the gradient via SGD):
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     # Update tracked metrics, including the loss metric, these are also configured in `compile()`:
    #     for metric in self.metrics:
    #         if metric.name == 'loss':
    #             metric.update_state(loss)
    #         else:
    #             metric.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value:
    #     return {m.name: m.result() for m in self.metrics}
    #
    # def test_step(self, data) -> Dict[str, Any]:
    #     """
    #     Overrides the :meth:`Model.test_step` method of the base class to customize what happens during calls to
    #     :meth:`Model.evaluate` during model evaluation. This method is not called during training (i.e. during
    #     evaluation on the validation set during fitting). Evaluation on validation metrics is handled in the
    #     :meth:`train_step` method.
    #
    #     See Also:
    #         - https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#providing_your_own_evaluation_step
    #         - https://docs.wandb.ai/tutorials/tensorflow_sweeps#%EF%B8%8F-build-a-simple-classifier-mlp
    #
    #     Args:
    #         data (Any): The data to train on. This could be a tuple of numpy arrays ``(x, y)`` if
    #           :meth:`Model.evaluate` was called with an ``x`` and ``y`` pair. If :meth:`Model.evaluate` was called with
    #           a TensorFlow dataset as in: ``model.evaluate(dataset, ...)`` then ``data`` will be what is yielded by the
    #           dataset iterator at each batch.
    #
    #     Returns:
    #
    #     """
    #     # Unpack the data:
    #     x, y = data
    #     # Compute predictions:
    #     y_pred = self(x, training=False)
    #     # Update the metrics:
    #     for metric in self.metrics:
    #         if metric.name != 'loss':
    #             metric.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value:
    #     return {m.name: m.result() for m in self.metrics}

    # def build(self, input_shape: Tuple[int, int, int], trial_hyperparameters: Dict[str, Any]) -> Model:
    #     # Build the model with the hyperparameters for this particular trial.
    #     # The concrete subclass should know which hyperparameters are pertinent to it for the particular trial.
    #     kernel_size = trial_hyperparameters['kernel_size']
    #     num_nodes_conv2d_1 = trial_hyperparameters['num_nodes_conv2d_1']
    #     optimizer = trial_hyperparameters['optimizer']
    #     # Construct the layers to append to this model (i.e. self):
    #     input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name='input_layer')
    #     conv_2d_1 = tf.keras.layers.Conv2D(
    #         filters=num_nodes_conv2d_1, kernel_size=kernel_size, activation='relu', name='conv_2d_1'
    #     )
    #     # Append the constructed layers:
    #     self.layers.append(input_layer)
    #     self.layers.append(conv_2d_1)
    #     # .. todo:: Delegate to super method to build the model?:
    #     super().build(input_shape=input_shape)
    #     # Compile the model:
    #     self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    #     # Return the model:
    #     return self

    # def save_and_upload_model(self, model_dir: str, wab_trial: Run, overwrite: bool):
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    #     model_path = os.path.join(model_dir, f"{wab_trial.trial_id}.h5")
    #     if os.path.exists(model_path):
    #         if overwrite:
    #             os.remove(model_path)
    #             super().save(model_path)
    #     else:
    #         super().save(model_path)
    #     # Upload the saved model to Weights and Biases:
    #     wab_trial.save(model_path)
    #

@tf.keras.utils.register_keras_serializable(name='MeanLayer')
class MeanLayer(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        # Mean along the time axis:
        result = tf.math.reduce_mean(inputs, axis=-2, keepdims=False)
        if len(result.shape) != 2:
            result = tf.expand_dims(result, axis=0)
        return result


@tf.keras.utils.register_keras_serializable(name='SummationLayer')
class SummationLayer(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        # Sum along the time axis:
        result = tf.math.reduce_sum(inputs, axis=-2, keepdims=False)
        if len(result.shape) != 2:
            result = tf.expand_dims(result, axis=0)
        return result


@tf.keras.utils.register_keras_serializable(name='MeanModel')
class MeanModel(Sequential):

    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, input_shape: Tuple[int, int, int],
            *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MeanModel'
        self._wab_trial_run = wab_trial_run
        self._trial_hyperparameters = trial_hyperparameters
        self._input_shape = input_shape
        super().__init__(*args, **kwargs)
        # Build the model with the hyperparameters for this particular trial. The concrete subclass should know which
        # hyperparameters are pertinent to it for the particular trial.
        # Construct the layers to append to this model (i.e. self) discard the channel dimension:
        self._input_layer = tf.keras.layers.InputLayer(input_shape=self._input_shape[:-1], name='input_layer')
        self._mean_layer = MeanLayer(name='mean_layer')
        self._output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        logger.debug(f"Constructing model...")
        self.add(self._input_layer)
        self.add(self._mean_layer)
        self.add(self._output_layer)
        self._repo_root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))

    def get_config(self):
        base_config = super().get_config()
        # .. todo:: Should we permit the wab_trial_run to be serialized?
        config = {
            'wab_trial_run': None, 'trial_hyperparameters': self._trial_hyperparameters.as_dict(),
            'input_shape': self._input_shape
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
            SummationModel: The model constructed from the provided configuration dictionary. Weights will be
            restored

        """
        logger.debug(f"from_config config: {config}")
        trial_hyperparameters_dict = config.pop('trial_hyperparameters')
        input_shape = config.pop('input_shape')
        layers = config.pop('layers')
        wab_trial_run = config.pop('wab_trial_run')
        trial_hyperparameters = Config()
        trial_hyperparameters.update(trial_hyperparameters_dict)
        return cls(wab_trial_run=None, trial_hyperparameters=trial_hyperparameters, input_shape=input_shape, **config)


@tf.keras.utils.register_keras_serializable(name='SummationModel')
class SummationModel(Sequential):

    def __init__(
            self, wab_trial_run: Optional[Run], trial_hyperparameters: Config, input_shape: Tuple[int, int, int],
            *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'SummationModel'
        self._wab_trial_run = wab_trial_run
        super().__init__(*args, **kwargs)
        self._trial_hyperparameters = trial_hyperparameters
        self._input_shape = input_shape
        # Build the model with the hyperparameters for this particular trial.
        # The concrete subclass should know which hyperparameters are pertinent to it for the particular trial.
        # Construct the layers to append to this model (i.e. self) discard the channel dimension:
        self._input_layer = tf.keras.layers.InputLayer(input_shape=self._input_shape[:-1], name='input_layer')
        self._summation_layer = SummationLayer(name='summation_layer')
        self._output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        logger.debug(f"Constructing model...")
        self.add(self._input_layer)
        self.add(self._summation_layer)
        self.add(self._output_layer)
        self._repo_root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))

    def get_config(self):
        base_config = super().get_config()
        # .. todo:: Should we permit the wab_trial_run to be serialized?
        config = {
            'wab_trial_run': None, 'trial_hyperparameters': self._trial_hyperparameters.as_dict(),
            'input_shape': self._input_shape
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
            SummationModel: The model constructed from the provided configuration dictionary. Weights will be
            restored

        """
        logger.debug(f"from_config config: {config}")
        trial_hyperparameters_dict = config.pop('trial_hyperparameters')
        input_shape = config.pop('input_shape')
        layers = config.pop('layers')
        wab_trial_run = config.pop('wab_trial_run')
        trial_hyperparameters = Config()
        trial_hyperparameters.update(trial_hyperparameters_dict)
        return cls(wab_trial_run=None, trial_hyperparameters=trial_hyperparameters, input_shape=input_shape, **config)


class PipingDetectorHyperModel:

    def __init__(
            self, train_ds: Dataset, val_ds: Optional[Dataset], test_ds: Dataset, num_classes: int, training: bool,
            hive_names_for_analysis: List[str], batch_size: int, resampled_train_steps_per_epoch: int,
            resampled_val_steps_per_epoch: int, metrics: List[Metric],
            wab_config_defaults: Optional[Dict[str, Any]] = None):
        """
        .. todo:: Docstrings.

        Args:
            train_ds (Dataset): The training dataset. If the training flag is set to ``False`` this is expected to be
              the training + validation datasets combined. Otherwise, this is the normal training dataset.
            val_ds (Optional[Dataset]): The validation dataset. If the training flag is set to ``False`` this is
              expected to be ``None`` (as the training and validation datasets will have been combined). Otherwise, this
              is the normal training set.
            test_ds (Dataset): The testing dataset.
            num_classes:
            training:
            hive_names_for_analysis:
            batch_size:
            resampled_train_steps_per_epoch:
            resampled_val_steps_per_epoch:
            metrics:
            wab_config_defaults:

        """
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._wab_config_defaults = wab_config_defaults
        logger.debug(f"train_ds.element_spec: {train_ds.element_spec}")
        self._image_shape = (train_ds.element_spec[0].shape[1], train_ds.element_spec[0].shape[2], train_ds.element_spec[0].shape[3])
        self._resampled_train_steps_per_epoch = resampled_train_steps_per_epoch
        self._resampled_val_steps_per_epoch = resampled_val_steps_per_epoch
        self._metrics = metrics
        logger.info(f"wab_sweep_config_defaults: {self._wab_config_defaults}")
        # Set default hyperparameter config values that remain fixed:
        # if self._wab_config_defaults is None:
        #     self._wab_config_defaults = {
        #         'image_size': (112, 112),
        #         'batch_size': 32
        #     }
        # logger.info(f"wab_sweep_config_defaults (post-init): {self._wab_config_defaults}")
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        self._training = training
        self._hive_names_for_analysis = hive_names_for_analysis
        # Generate a unique group for this run:
        hive_names_shorthand = [self.hive_name_to_shorthand(hive_name) for hive_name in self._hive_names_for_analysis]
        self._wab_group_name = f"{''.join(hive_names_shorthand)}-{wab.util.generate_id()}"

    @staticmethod
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

    def construct_model_run_trial(self):
        # Initialize the namespace/container for this particular trial run with WandB:
        wab_trial_run = wab.init(
            project='PipingDetection', entity='appmais', config=wab.config, group=self._wab_group_name
        )
        # Workaround for exception logging:
        sys.excepthook = exc_handler
        # Wandb agent will override the defaults with the sweep configuration subset it has selected according to the
        # specified 'method' in the config:
        logger.info(f"wandb.config: {wab.config}")
        model = PipingDetectorWabModel(
            wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape,
            name='piping_detector'
        )
        # model = SummationModel(
        #     wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape,
        #     name='summation_model'
        # )
        # model = MeanModel(
        #     wab_trial_run=wab_trial_run, trial_hyperparameters=wab.config, input_shape=self._image_shape,
        #     name='mean_model'
        # )
        optimizer_config = wab.config['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_learning_rate = optimizer_config['learning_rate']
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=optimizer_learning_rate)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type} provided in the hyperparameter section of the "
                         f"sweep configuration.")
            exit(1)
        # .. todo:: Batch size should be known now? Is that why output of the layer is multiple in model.summary()?
        model.build(input_shape=(self._batch_size, *self._image_shape))
        # compile the model:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=self._metrics)
        # .. todo: Should hparams be part of build or constructor?
        # model = model.build(input_shape=self._image_shape, trial_hyperparameters=wab.config)
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
                val_ds=self._val_ds,
                num_epochs=wab.config['num_epochs'],
                resampled_train_steps_per_epoch=self._resampled_train_steps_per_epoch,
                resampled_val_steps_per_epoch=self._resampled_val_steps_per_epoch,
                validation_batch_size=self._batch_size,
                inference_target_conv_layer_name=wab.config['inference_target_conv_layer_name']
            )
        else:
            # Support for final training run performed after model selection process:
            self.run_trial(
                model=model, num_classes=self._num_classes, wab_trial_run=wab_trial_run,
                train_ds=self._train_ds, val_ds=self._test_ds,
                num_epochs=wab.config['num_epochs'],
                resampled_train_steps_per_epoch=self._resampled_train_steps_per_epoch,
                resampled_val_steps_per_epoch=self._resampled_val_steps_per_epoch,
                validation_batch_size=self._batch_size,
                inference_target_conv_layer_name=wab.config['inference_target_conv_layer_name']
            )
        wab_trial_run.finish()
        tf.keras.backend.clear_session()

    @staticmethod
    def run_trial(
            model: Model, num_classes: int, wab_trial_run: Run, train_ds: Dataset, val_ds: Dataset, num_epochs: int,
            resampled_train_steps_per_epoch: int, resampled_val_steps_per_epoch: int, validation_batch_size: int,
            inference_target_conv_layer_name: str) -> History:
        """

        Args:
            model:
            wab_trial_run:
            train_ds:
            val_ds:
            num_epochs:
            resampled_train_steps_per_epoch:
            resampled_val_steps_per_epoch:
            validation_batch_size:
            inference_target_conv_layer_name (str): The name of the Conv2D layer to run Grad-CAM related inference
              visualizations on.

        Returns:

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
        confusion_matrix_callback = ConfusionMatrixCallback(
            num_classes=2, wab_trial_run=wab_trial_run, validation_data=val_ds,
            validation_steps=resampled_val_steps_per_epoch
        )
        grad_cam_callback = GradCAMCallback(
            num_images=5, num_classes=num_classes, wab_trial_run=wab_trial_run, validation_data=val_ds,
            validation_batch_size=validation_batch_size, validation_steps=resampled_val_steps_per_epoch,
            target_conv_layer_name=inference_target_conv_layer_name, log_conv2d_output=True, log_grad_cam_heatmaps=True,
            log_target_conv_layer_kernels=True, grad_cam_heatmap_alpha_value=0.4
        )
        trial_history = model.fit(
            train_ds, validation_data=val_ds, epochs=num_epochs, steps_per_epoch=resampled_train_steps_per_epoch,
            validation_steps=resampled_val_steps_per_epoch,
            callbacks=[wab_callback, confusion_matrix_callback, grad_cam_callback]
            # callbacks=[
            #     wab_callback, confusion_matrix_callback, grad_cam_callback
            # ]
        )
        return trial_history

    # def run_trial(self, train_ds: Dataset, val_ds: Dataset, num_epochs: int, trial_hyperparameters: Dict[str, Any]) -> History:
    #     """
    #
    #     .. todo:: Note that train_ds may actually be (train_ds + val_ds) and val_ds may really be test_ds during the
    #         final training run after model selection has occurred.
    #
    #     Args:
    #         train_ds:
    #         val_ds:
    #         trial_hyperparameters:
    #
    #     Returns:
    #
    #     """
    #     # Initialize the namespace/container for this particular trial with WandB:
    #     wab_trial_run = wab.init(
    #         project=self._wab_config['project'],
    #         config=trial_hyperparameters,
    #         group=self._wab_group_name
    #     )
    #     # Builds the model with the provided hyperparameters and trains it.
    #     # Initialize the model with the hyperparameters for this particular trial:
    #     self._model = PipingDetectorWabModel()
    #     # Build the model with the hyperparameters for this particular trial:
    #     self._model = self._model.build(
    #         input_shape=self._input_image_shape,
    #         trial_hyperparameters=trial_hyperparameters
    #     )
    #     # Train the model:
    #     # .. todo:: Early stopping callbacks?
    #     trial_history = self._model.fit(
    #         train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[WandbCallback()]
    #     )
    #     # .. todo:: Model serialization and upload to WandB
    #     # .. todo:: This could be where model interpretability visualizations are created and uploaded to WandB. But
    #     #     perhaps this is best done with a WanbEvalCallback?
    #     return trial_history


@tf.function
def preprocess_image_path(image_path: str, label: int, image_shape: Tuple[int, int, int], downsample_image: bool) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
        image_shape (Tuple[int, int, int]): The native size of the input image, that is, the size of the input image
          prior to downsampling (if requested). Expected to be of the form ``(width, height, channels)``.
        downsample_image (bool): A boolean flag indicating if the provided image should be downsampled to half of the
          specified ``image_shape``.

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
    image = tf.image.decode_png(raw_byte_string, channels=image_shape[-1])
    image = tf.image.convert_image_dtype(image, tf.float32)
    if downsample_image:
        downsampled_image_size = (image_shape[0] // 2, image_shape[1] // 2)
        image = tf.image.resize(image, downsampled_image_size)
        image.set_shape(downsampled_image_size + (image_shape[-1],))
    else:
        image.set_shape((image_shape[0], image_shape[1], image_shape[-1]))
    return image, label


def load_datasets_for_hive(
        dataset_split: DatasetSplit, hive_name: str, hive_dataset_split_root_dir: str, seed: int,
        image_shape: Tuple[int, int, int], downsample_image: bool) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Constructs a TensorFlow dataset for the specified hive which contains a list of file paths to the images in the
    provided ``dataset_split``.

    Warnings:
        This method currently discards half of the indexed file paths for the ``DatasetSplit.TRAIN`` dataset split only.
        This is done to speed up training (no discernible loss in accuracy). Half of the file paths associated with the
        positive class are discarded, as well as half the samples associated with the negative class.

    Args:
        dataset_split (DatasetSplit): The dataset split to load for the specified hive. Only images in the subdirectory
          corresponding to the provided ``dataset_split`` will be loaded.
        hive_name (str): The hive name for which to load the specified dataset split.
        hive_dataset_split_root_dir (str): The containing subdirectory for the specified ``dataset_split``. This should
          be a directory under the provided ``hive_name`` directory.
        seed (int): The random seed to (hopefully) ensure reproducibility in how TensorFlow indexes the subdirectories
          for the provided ``hive_name`` and ``dataset_split``.
        image_shape (Tuple[int, int, int]): The native size of the input image, that is, the size of the input image
          prior to any requested down-sampling (which happens later during preprocessing). Expected to be of the form
          ``(width, height, channels)``.
        downsample_image (bool): A boolean value indicating if the loaded images should be dynamically downsampled to
          ``1/2`` the provided native resolution specified by ``image_shape``.

    Returns:
        Tuple[Dataset, Dataset, Dataset, Dataset]: A size-4 tuple containing a TensorFlow dataset of file paths for the
        specified ``hive_name`` and ``dataset_split``. Data is separated into a list of file paths that correspond to
        positive and negative class labels. This is done for ease in downstream oversampling.

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
    split_ds_neg_files = Dataset.list_files(negative_split_data_dir + '/*.png', seed=seed)
    # Remove half of the training data:
    if dataset_split == DatasetSplit.TRAIN:
        logger.warning(f"Removing half of the negative {dataset_split} data for hive {hive_name}.")
        split_ds_neg_files = split_ds_neg_files.take(num_negative_split_samples // 2)
        logger.debug(f"Identified {split_ds_neg_files.cardinality().numpy()} negative {dataset_split} samples for hive "
                     f"{hive_name}.")
    label = tf.constant(0, dtype=tf.int32)
    split_ds_neg = split_ds_neg_files.map(partial(preprocess_image_path, label=label, image_shape=image_shape,
                                                  downsample_image=downsample_image))
    # Load positive label split data:
    split_ds_pos_files = tf.data.Dataset.list_files(positive_split_data_dir + '/*.png', seed=seed)
    # Remove half of the training data:
    if dataset_split == DatasetSplit.TRAIN:
        logger.warning(f"Removing half of the positive {dataset_split} data for hive {hive_name}.")
        split_ds_pos_files = split_ds_pos_files.take(num_positive_split_samples // 2)
        logger.debug(f"Identified {split_ds_pos_files.cardinality().numpy()} positive {dataset_split} samples for hive "
                     f"{hive_name}.")
    label = tf.constant(1, dtype=tf.int32)
    split_ds_pos = split_ds_pos_files.map(partial(preprocess_image_path, label=label, image_shape=image_shape,
                                                  downsample_image=downsample_image))
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


def aggregate_datasets_across_hives(
        hive_names: List[str], root_data_dir: str, seed: int, image_shape: Tuple[int, int, int],
        downsample_image: bool):
    """
    This method steps over the provided list of hives and aggregates the positive and negative class samples for each
    hive into a single TensorFlow dataset.

    Args:
        hive_names (List[str]): The names of ht
        root_data_dir (str): The parent folder which contains subdirectories for each hive.
        seed (int): The random seed to pass downstream for use in the file path constructors in
          :meth:`load_datasets_for_hive`.
        image_shape (Tuple[int, int, int]): The native size of the input image, that is, the size of the input image
          prior to any requested down-sampling. Expected to be of the form ``(width, height, channels)``.
        downsample_image (bool): A boolean value indicating if the loaded images should be dynamically downsampled to
            ``1/2`` the provided native resolution specified by ``image_shape``.

    Returns:
        Tuple[Dataset, ..., Dataset]: A Tuple of size 12 which contains a TensorFlow dataset for each of the dataset
        splits (train, val, and test) separated by positive and negative class samples. The returned datasets have been
        aggregated across hives.

    """
    train_ds_neg_files, train_ds_neg, train_ds_pos_files, train_ds_pos = None, None, None, None
    val_ds_neg_files, val_ds_neg, val_ds_pos_files, val_ds_pos = None, None, None, None
    test_ds_neg_files, test_ds_neg, test_ds_pos_files, test_ds_pos = None, None, None, None
    logger.debug(f"Aggregating pre-split data for hives {hive_names}.")
    # First load each hive's individual (probably unbalanced) datasets:
    for i, hive_name in enumerate(hive_names):
        hive_data_dir = os.path.join(root_data_dir, hive_name)
        assert os.path.exists(hive_data_dir), f"Could not find data directory for hive {hive_name} at {hive_data_dir}"
        # Training data for this hive:
        hive_dataset_train_dir = os.path.join(hive_data_dir, 'train')
        hive_train_ds_neg_files, hive_train_ds_neg, hive_train_ds_pos_files, hive_train_ds_pos = (
            load_datasets_for_hive(
                dataset_split=DatasetSplit.TRAIN,
                hive_name=hive_name,
                hive_dataset_split_root_dir=hive_dataset_train_dir,
                seed=seed,
                image_shape=image_shape,
                downsample_image=downsample_image
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
            load_datasets_for_hive(
                dataset_split=DatasetSplit.VALIDATION,
                hive_name=hive_name,
                hive_dataset_split_root_dir=hive_dataset_val_dir,
                seed=seed,
                image_shape=image_shape,
                downsample_image=downsample_image
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
            load_datasets_for_hive(
                dataset_split=DatasetSplit.TEST,
                hive_name=hive_name,
                hive_dataset_split_root_dir=hive_dataset_test_dir,
                seed=seed,
                image_shape=image_shape,
                downsample_image=downsample_image
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
        dataset_split: DatasetSplit, split_ds_neg: Dataset, split_ds_pos: Dataset,
        split_ds_neg_files: Dataset, split_ds_pos_files: Dataset, batch_size: int, seed: int):
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
        batch_size (int): The batch size to use for the returned oversampled datasets.
        seed (int): The random seed to use for the returned oversampled datasets.
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
        buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
    ).repeat()
    split_ds_pos = split_ds_pos.shuffle(
        buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
    ).repeat()
    split_ds = Dataset.sample_from_datasets(
        datasets=[split_ds_pos, split_ds_neg],
        weights=[0.5, 0.5],
        seed=seed
    )
    # .. todo:: The 2 here is hardcoded because we are doing a 50/50 split of positive and negative examples.
    #     This would need to be changed if we were to do a different split.
    # .. todo:: The resampled_steps_per_epoch should be dependent on the majority class label, not necessarily only
    #     the negative class.
    resampled_steps_per_epoch = int(np.ceil(num_negative_split_samples / (batch_size / 2)))
    # Construct oversampled dataset of files:
    split_ds_neg_files = split_ds_neg_files.shuffle(
        buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
    ).repeat()
    split_ds_pos_files = split_ds_pos_files.shuffle(
        buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
    ).repeat()
    split_ds_files = Dataset.sample_from_datasets(
        datasets=[split_ds_pos_files, split_ds_neg_files],
        weights=[0.5, 0.5],
        seed=seed
    )
    # Shuffle and pre-batch datasets:
    split_ds = split_ds.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False)
    split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=True)
    # split_ds = split_ds.cache()
    split_ds_files = split_ds_files.shuffle(
        buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
    )
    split_ds_files = split_ds_files.batch(batch_size=batch_size, drop_remainder=True)
    # split_ds_files = train_ds_files.cache()
    return split_ds_files, split_ds, resampled_steps_per_epoch


def load_datasets(
        root_data_dir: str, seed: int, batch_size: int, image_shape: Tuple[int, int, int], downsample_image: bool,
        upload_to_wandb: bool, combine_training_and_validation_sets: bool, hive_names: Optional[List[str]] = None,
        over_sample_train_set_positive_class: bool = True, over_sample_val_set_positive_class: bool = False,
        over_sample_test_set_positive_class: bool = False) \
        -> Tuple[Dataset, Dataset, Dataset, Optional[int], Optional[int], Optional[int]]:
    """
    This method handles the loading and weighting of the training, validation, and testing datasets. Data will be
    loaded from the hives specified in the list of provided ``hives_names``. If no ``hive_names`` are provided then
    the data from all hives will be leveraged.

    Args:
        root_data_dir (str): The root directory where the training data images are located.
        seed (int): The random seed to use for shuffling the dataset.
        batch_size (int): How many images should be included in a batch. TensorFlow datasets are pre-batched.
        image_shape (Tuple[int, int, int]): The *native* shape of the images to be loaded (prior to any requested
          down-sampling). This should be a 3-tuple in the form ``(width, height, channels)``.
        downsample_image (bool): A boolean value indicating if the loaded images should be dynamically downsampled to
          ``1/2`` the provided native resolution specified by ``image_shape``.
        upload_to_wandb (bool): Indicates whether training data should be uploaded directly to WandB. This may slow
          down :class:`tf.data.Dataset` loading, but can be helpful for debugging.
        combine_training_and_validation_sets (bool): A boolean indicating if the training and validation datasets should
          be combined into a single dataset. This is useful for the final round of evaluation (e.g. model performance
          instead of model selection). If this flag is set to ``True`` then the returned validation dataset will be
          ``None``. Additionally, the ``over_sample_val_set_positive_class`` flag will be ignored (and the
          ``over_sample_train_set_positive_class`` flag will take precedence as a stand-in for both sets). Additionally,
          if this flag is set to ``True`` then the returned ``resampled_val_steps_per_epoch`` will be ``None`` (and the
          returned ``resampled_train_steps_per_epoch`` will be used for both sets).
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
    assert os.path.exists(root_data_dir), f"Could not find root data directory at {root_data_dir}"

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
        aggregate_datasets_across_hives(
            hive_names=hive_names,
            root_data_dir=root_data_dir,
            seed=seed,
            image_shape=image_shape,
            downsample_image=downsample_image
        )
    )
    if combine_training_and_validation_sets:
        logger.warning(f"Combining training and validation sets into a single dataset.")
        train_ds_neg_files = train_ds_neg_files.concatenate(val_ds_neg_files)
        train_ds_pos_files = train_ds_pos_files.concatenate(val_ds_pos_files)
        train_ds_neg = train_ds_neg.concatenate(val_ds_neg)
        train_ds_pos = train_ds_pos.concatenate(val_ds_pos)
        # Remove the validation set (it is now part of the training set):
        val_ds_neg_files = None
        val_ds_pos_files = None
        val_ds_neg = None
        val_ds_pos = None
    if over_sample_train_set_positive_class:
        train_ds_files, train_ds, resampled_train_steps_per_epoch = get_oversampled_dataset(
            dataset_split=DatasetSplit.TRAIN,
            split_ds_neg=train_ds_neg,
            split_ds_pos=train_ds_pos,
            split_ds_neg_files=train_ds_neg_files,
            split_ds_pos_files=train_ds_pos_files,
            batch_size=batch_size,
            seed=seed
        )
    else:
        train_ds = Dataset.concatenate(train_ds_pos, train_ds_neg)
        train_ds = train_ds.shuffle(
            buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
        )
        train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
        # train_ds = train_ds.cache()
        train_ds_files = Dataset.concatenate(train_ds_pos_files, train_ds_neg_files)
        train_ds_files = train_ds_files.shuffle(
            buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
        )
        train_ds_files = train_ds_files.batch(batch_size=batch_size, drop_remainder=True)
        # train_ds_files = train_ds_files.cache()
        resampled_train_steps_per_epoch = None
    # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ''' Load the validation data: '''
    if not combine_training_and_validation_sets:
        # If we are not combining the training and validation sets, then we need to load the validation data:
        if over_sample_val_set_positive_class:
            val_ds_files, val_ds, resampled_val_steps_per_epoch = get_oversampled_dataset(
                dataset_split=DatasetSplit.VALIDATION,
                split_ds_neg=val_ds_neg,
                split_ds_pos=val_ds_pos,
                split_ds_neg_files=val_ds_neg_files,
                split_ds_pos_files=val_ds_pos_files,
                batch_size=batch_size,
                seed=seed
            )
        else:
            val_ds = Dataset.concatenate(val_ds_pos, val_ds_neg)
            val_ds = val_ds.shuffle(
                buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
            )
            val_ds = val_ds.batch(batch_size=batch_size, drop_remainder=True)
            # val_ds = val_ds.cache()
            val_ds_files = Dataset.concatenate(val_ds_pos_files, val_ds_neg_files)
            val_ds_files = val_ds_files.shuffle(
                buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
            )
            val_ds_files = val_ds_files.batch(batch_size=batch_size, drop_remainder=True)
            # val_ds_files = val_ds_files.cache()
            resampled_val_steps_per_epoch = None
        # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        # val_ds_files = val_ds_files.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        # Training and validation sets are combined.
        val_ds = None
        resampled_val_steps_per_epoch = None
    ''' Load the testing data: '''
    if over_sample_test_set_positive_class:
        test_ds_files, test_ds, resampled_test_steps_per_epoch = get_oversampled_dataset(
            dataset_split=DatasetSplit.TEST,
            split_ds_neg=test_ds_neg,
            split_ds_pos=test_ds_pos,
            split_ds_neg_files=test_ds_neg_files,
            split_ds_pos_files=test_ds_pos_files,
            batch_size=batch_size,
            seed=seed
        )
    else:
        test_ds = Dataset.concatenate(test_ds_pos, test_ds_neg)
        test_ds = test_ds.shuffle(
            buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
        )
        test_ds = test_ds.batch(batch_size=batch_size, drop_remainder=True)
        # test_ds = test_ds.cache()
        test_ds_files = Dataset.concatenate(test_ds_pos_files, test_ds_neg_files)
        test_ds_files = test_ds_files.shuffle(
            buffer_size=batch_size, seed=seed, reshuffle_each_iteration=False
        )
        test_ds_files = test_ds_files.batch(batch_size=batch_size, drop_remainder=True)
        # test_ds_files = test_ds_files.cache()
        resampled_test_steps_per_epoch = None
    # .. todo:: May not want to prefetch-autotune see: https://www.tensorflow.org/guide/data_performance
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return (train_ds, val_ds, test_ds, resampled_train_steps_per_epoch, resampled_val_steps_per_epoch,
            resampled_test_steps_per_epoch)


def main():
    # Note: Do not use the keras object for the optimizer (e.g. Adam(learning_rate=0.001) instead of 'adam')
    # or you get a RuntimeError('Should only create a single instance of _DefaultDistributionStrategy.')
    hyperparameters = {
        'num_epochs': {
            'value': 1600
            # 'values': [400]
        },
        'conv_layer_activation_function': {
            'value': 'tanh'
        },
        'kernel_size': {
            'value': 11
        },
        'num_nodes_conv_1': {
            'value': 2**3
        },
        'num_nodes_conv_2': {
            'value': 2**0
        },
        'optimizer': {
            'parameters': {
                'type': {
                    'value': 'adam',
                    # 'values': ['adam', 'rmsprop']
                },
                'learning_rate': {
                    'values': [0.001]
                }
            }
        },
        'inference_target_conv_layer_name': {
            'values': ['conv_2d_2']
        }
    }
    sweep_configuration = {
        'method': 'grid',
        # 'name': 'sweepy',
        'project': 'PipingDetection',
        'entity': 'appmais',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        # 'early_terminate': {
        #     'type': 'hyperband',
        #     'min_iter': 5
        # },
        'parameters': hyperparameters
    }
    sweep_id = wab.sweep(sweep=sweep_configuration, project='PipingDetection', entity='appmais')
    '''
    Initialize a TensorFlow dataset from disk.
    See Also:
        https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    '''
    # Initialize TensorFlow datasets from disk:
    data_root_dir = os.path.abspath(f'/home/bee/piping-detection/split_spectra_images')
    train_ds, val_ds, test_ds, resampled_train_steps_per_epoch, resampled_val_steps_per_epoch, _ = (
        load_datasets(
            root_data_dir=data_root_dir,
            upload_to_wandb=False,
            hive_names=HIVE_NAMES_FOR_ANALYSIS,
            over_sample_train_set_positive_class=True,
            over_sample_val_set_positive_class=True,
            over_sample_test_set_positive_class=False,
            combine_training_and_validation_sets=True,
            batch_size=BATCH_SIZE,
            seed=SEED,
            image_shape=IMAGE_SHAPE,
            downsample_image=DOWNSAMPLE_IMAGES
        )
    )
    hypermodel = PipingDetectorHyperModel(
        wab_config_defaults=None, num_classes=NUM_CLASSES, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        training=False,
        batch_size=BATCH_SIZE,
        hive_names_for_analysis=HIVE_NAMES_FOR_ANALYSIS,
        resampled_train_steps_per_epoch=resampled_train_steps_per_epoch,
        resampled_val_steps_per_epoch=resampled_val_steps_per_epoch,
        metrics=[
            'accuracy', 'binary_accuracy', tf.keras.metrics.BinaryCrossentropy(from_logits=False),
            tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
        ]
    )
    # wab.agent(
    #     count=NUM_TRIALS, project='PipingDetection', entity='appmais', sweep_id=sweep_id,
    #     function=hypermodel.construct_model_run_trial
    # )
    wab.agent(
        count=NUM_TRIALS, sweep_id=sweep_id, project="PipingDetection", entity="appmais",
        function=hypermodel.construct_model_run_trial
    )


def exc_handler(exc_type, exc, tb):
    logger.exception(f"EXCEPTION")
    print("EXCEPTION")
    traceback.print_exception(exc_type, exc, tb)

if __name__ == '__main__':
    NUM_TRIALS = 10
    HIVE_NAMES_FOR_ANALYSIS = ['AppMAIS2L', 'AppMAIS7L', 'AppMAIS7R']
    BATCH_SIZE = 512
    NUM_CLASSES = 2
    SEED = 42
    tf.random.set_seed(seed=SEED)
    # Images were exported as grayscale images and downsampled to 112 x 112:
    IMAGE_SHAPE = (224, 224, 1)
    DOWNSAMPLE_IMAGES = True
    REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    logger.debug(f"REPO_ROOT_DIR: {REPO_ROOT_DIR}")
    WANDB_LOG_DIR = REPO_ROOT_DIR
    if not os.path.exists(WANDB_LOG_DIR):
        os.makedirs(WANDB_LOG_DIR)
    os.environ['WANDB_DIR'] = WANDB_LOG_DIR
    logger.info(f'WANDB_DIR: {WANDB_LOG_DIR}')
    main()
    # Better error handling by catching exceptions prior to wandb sweep agent terminating:
    # try:
    #     main()
    # except Exception as err:
    #     # logger.exception(f"{traceback.print_exc()}", file=sys.stderr)
    #     print(traceback.print_exc(), file=sys.stderr)
    #     exit(1)
    # finally:
    #     wandb.finish()

# def main():
#     # Note: Do not use the keras object for the optimizer (e.g. Adam(learning_rate=0.001) instead of 'adam')
#     # or you get a RuntimeError('Should only create a single instance of _DefaultDistributionStrategy.')
#     pass

# if __name__ == '__main__':
#     """
#     Main entry point for the sweeper. This script is used to run the hyperparameter search. Define the hyperparameteres
#     which will remain constant between runs below. The hyperparameters which will be varied should be defined in the
#     sweep configuration.
#     """
#     NUM_TRIALS = 10
#     BATCH_SIZE = 25
#     NUM_CLASSES = 2
#     SEED = 42
#     REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
#     LOG_DIR = os.path.join(REPO_ROOT_DIR, 'logs')
#     if not os.path.exists(LOG_DIR):
#         os.makedirs(LOG_DIR)
#     os.environ['WANDB_DIR'] = LOG_DIR
#     LOCAL_DATA_DIR = os.path.join(REPO_ROOT_DIR, 'data')
#     if not os.path.exists(LOCAL_DATA_DIR):
#         os.makedirs(LOCAL_DATA_DIR)
#     DATA_DIR = '/usr/local/data/JustRAIGS/raw/'
#     logger.debug(f"WANDB_DIR: {LOG_DIR}")
#     logger.debug(f"LOCAL_DATA_DIR: {LOCAL_DATA_DIR}")
#     logger.debug(f"DATA_DIR: {DATA_DIR}")
#
#
#     tf.random.set_seed(seed=SEED)


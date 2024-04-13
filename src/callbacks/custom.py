import numpy as np
from loguru import logger
import seaborn as sns
from typing import Optional
from matplotlib import cm
from matplotlib import colors
from wandb.apis.public import Run
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb as wab


# @tf.keras.utils.register_keras_serializable(name='ConfusionMatrixCallback')
class ConfusionMatrixCallback(Callback):
    """
    Plots the confusion matrix on the validation dataset after training has finished and uploads the results to WandB.

    Notes::
      This class assumes that the validation dataset is a TensorFlow :class:`tf.data.Dataset` object that yields (x, y)
      tuples of validation data. This Callback assumes a binary classification problem. If this is not the case, the
      callback would need to be modified to handle the multi-class classification problem.

    See Also:
        - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
        - https://docs.wandb.ai/guides/integrations/keras

    """

    def __init__(
            self, num_classes: int, wab_trial_run: Run, validation_data: Dataset, epoch_frequency: int,
            validation_steps: Optional[int] = None):
        """

        Args:
            num_classes (int): The number of classes for the classification problem.
            wab_trial_run (Run): An instance of the WandB Run class for the current trial.
            validation_data (Dataset): A tensorflow Dataset object containing the validation data. This dataset may or
              may not have infinite cardinality at runtime (as a result of oversampling). The dataset will yield (x, y)
              tuples of validation data.
            epoch_frequency (int): The frequency (in epochs) at which the confusion matrix should be generated and
              logged to WandB.
            validation_steps (Optional[int]): The number of steps/batches to be considered one epoch for the validation
              dataset. This value must be provided if the validation dataset has infinite cardinality at runtime.

        """
        self._num_classes = num_classes
        if self._num_classes > 2:
            self._is_multiclass_classification = True
        else:
            self._is_multiclass_classification = False
        self._wab_trial_run = wab_trial_run
        self._validation_data = validation_data
        self._epoch_frequency = epoch_frequency
        self._validation_steps = validation_steps
        super().__init__()

    def _multilabel_confusion_matrix(self, epoch: int):
        logger.debug(f"Generating confusion matrix for {self._num_classes} classes on the validation dataset...")
        # Check to see if the provided validation dataset is infinite:
        if self._validation_data.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
            if self._validation_steps is None:
                raise ValueError("Since the provided validation dataset is infinite, the number of validation steps "
                                 "must be specified.")
        index_to_label_map = {0: 'ANRS', 1: 'ANRI', 2: 'RNFLDS', 3: 'RNFLDI', 4: 'BCLVS', 5: 'BCLVI', 6: 'NVT', 7: 'DH', 8: 'LD', 9: 'LC'}
        y_pred_prob = []
        y_true = []
        y_pred_bool = []
        for i, batch in enumerate(self._validation_data):
            if self._validation_steps is not None:
                if i == self._validation_steps:
                    break
            images = batch[0]
            labels = batch[1]
            for j, (image, true_labels) in enumerate(zip(images.numpy(), labels.numpy())):
                y_true.append(list(true_labels))
            y_pred_batch = self.model.predict_on_batch(images).squeeze()
            for j, (image, pred_probs) in enumerate(zip(images, y_pred_batch)):
                y_pred_prob.append(list(pred_probs))
                y_pred_bool.append([True if pred_prob > 0.5 else False for pred_prob in pred_probs])
        '''
        Confusion matrix in the multiclass case (see 
        https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/metrics/multilabel_confusion_matrix.py#L28-L188
        ):
        '''
        y_true = np.array(y_true, dtype=np.bool)
        y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
        y_pred_bool = np.array(y_pred_bool, dtype=np.bool)
        # y_true = tf.cast(y_true, tf.int32)
        # true positive
        true_positive = tf.math.count_nonzero(y_true * y_pred_bool, 0)
        # predictions sum
        pred_sum = tf.math.count_nonzero(y_pred_bool, 0)
        # true labels sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive
        y_true_negative = tf.math.not_equal(y_true, True)
        y_pred_negative = tf.math.not_equal(y_pred_bool, True)
        true_negative = tf.math.count_nonzero(
            tf.math.logical_and(y_true_negative, y_pred_negative), axis=0
        )
        # Flatten confusion matrix:
        confusion_matrix = tf.convert_to_tensor(
            [true_negative, false_positive, false_negative, true_positive], dtype=tf.int32
        )
        # Reshape into 2*2 matrix:
        confusion_matrix = tf.reshape(tf.transpose(confusion_matrix), [-1, 2, 2])

        min_count = 0
        max_count = tf.reduce_max(confusion_matrix)
        normalizer = colors.Normalize(vmin=min_count, vmax=max_count)
        # cbar_scalar_mappable = cm.ScalarMappable(norm=normalizer, cmap=sns.color_palette("rocket", as_cmap=True))
        fig = plt.figure(figsize=(40, 5))
        cax = fig.add_subplot(1, 11, 11)
        axes = []
        for i in range(self._num_classes):
            # ax = axes[i]
            if i == 0:
                ax = fig.add_subplot(1, 11, i+1)
                sns.heatmap(
                    confusion_matrix[i], ax=ax, cbar_ax=cax, cbar_kws={
                        'norm': normalizer
                    }, annot=True, fmt="g", cbar=True, xticklabels=['Neg.', 'Pos.'],
                    yticklabels=['Neg.', 'Pos.']
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_xticklabels(['Neg.', 'Pos.'], rotation=90)
                ax.set_yticklabels(['Neg.', 'Pos.'], rotation=0)
            else:
                ax = fig.add_subplot(1, 11, i+1, sharex=axes[i-1], sharey=axes[i-1])
                sns.heatmap(
                    confusion_matrix[i], ax=ax, cbar_ax=cax, annot=True, fmt="g", cbar=True, xticklabels=True,
                    yticklabels=False, cbar_kws={
                        'norm': normalizer
                    }
                )
                ax.set_xlabel('Predicted')
                ax.set_xticklabels(['Neg.', 'Pos.'], rotation=90)
                # ax.set_yticklabels(['Neg.', 'Pos.'], rotation=0)
            ax.set_title(f"{index_to_label_map[i]}")
            axes.append(ax)
        plt.tight_layout()
        # plt.suptitle(f"Confusion Matrix at Epoch {epoch}")
        # plt.show()
        self._wab_trial_run.log({'confusion_matrix': wab.Image(fig)})
        plt.clf()
        plt.close(fig)

    def _binary_confusion_matrix(self):
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
            class_names_batch = np.array(['RG' if int(label.numpy()) == 1 else 'NRG' for label in labels])
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

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._epoch_frequency == 0:
            logger.debug(f"Logging confusion matrix at epoch {epoch} at a frequency of {self._epoch_frequency} epochs.")
            if self._is_multiclass_classification:
                self._multilabel_confusion_matrix(epoch=epoch)
            else:
                self._binary_confusion_matrix()

    def on_train_end(self, logs=None):
        """
        Called by the Keras framework at the end of Model training. This method is responsible for computing the
        Confusion Matrix on the validation dataset provided during callback initialization, and uploading the results
        to WaB.

        """
        self._confusion_matrix()


class TrainValImageCallback(Callback):
    """
    Plots a few images from the training and validation datasets and uploads the results to WandB.
    Each image has a title of either 'RG' or 'NRG' depending on the class label.
    """

    def __init__(self, wab_trial_run: Run, train_data: Dataset, val_data: Dataset, num_images: int):
        """

        Args:
            wab_trial_run (Run): An instance of the WandB Run class for the current trial.
            train_data (Dataset): A tensorflow Dataset object containing the training data. This dataset may or may not
            have infinite cardinality at runtime (as a result of oversampling). The dataset will yield (x, y) tuples of
            training data.
            val_data (Dataset): A tensorflow Dataset object containing the validation data. This dataset may or may not
            have infinite cardinality at runtime (as a result of oversampling). The dataset will yield (x, y) tuples of
            validation data.
            num_images (int): The number of images to plot from the training and validation datasets.

        """
        self._wab_trial_run = wab_trial_run
        self._train_data = train_data
        self._val_data = val_data
        assert num_images > 0, "The number of images to plot must be greater than 0."
        self._num_images = num_images
        super().__init__()

    def on_train_end(self, logs=None):
        """
        Called by the Keras framework at the end of Model training. This method is responsible for plotting a few images
        from the training and validation datasets provided during callback initialization, and uploading the results to
        WaB.

        """
        logger.debug(f"Plotting {self._num_images} images from the training and validation datasets...")
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        for i, batch in enumerate(self._train_data):
            images = batch[0]
            labels = batch[1]
            train_images.extend(images)
            train_labels.extend(labels)
            if len(train_images) >= self._num_images:
                break
        for i, batch in enumerate(self._val_data):
            images = batch[0]
            labels = batch[1]
            val_images.extend(images)
            val_labels.extend(labels)
            if len(val_images) >= self._num_images:
                break
        fig, axs = plt.subplots(2, self._num_images)
        for i in range(self._num_images):
            axs[0, i].imshow(np.ndarray.astype(train_images[i].numpy() * 255, np.uint8))
            # Will definitely need to be changed for this to work with multi-class classification problems:
            axs[0, i].set_title('RG' if train_labels[i].numpy() == 1 else 'NRG')
            axs[0, i].axis('off')
            axs[1, i].imshow(np.ndarray.astype(val_images[i].numpy() * 255, np.uint8))
            axs[1, i].set_title('RG' if val_labels[i].numpy() == 1 else 'NRG')
            axs[1, i].axis('off')
        # Label the first row of images as training images and the second row as validation images:
        axs[0, 0].set_ylabel('Training Images')
        axs[1, 0].set_ylabel('Validation Images')
        plt.tight_layout()
        self._wab_trial_run.log({'train_val_images': wab.Image(fig)})
        plt.clf()
        plt.close(fig)


class GradCAMCallback(Callback):
    """
    Runs Grad-CAM on a subset of random images from the validation set (where the predicted class label is correct) and
    uploads the results to WandB. This callback is designed to execute only when training is finished.
    """

    def __init__(
            self, num_images: int, num_classes: int, wab_trial_run: Run, target_conv_layer_name: str,
            validation_data: Dataset, epoch_frequency: int, validation_steps: Optional[int] = None,
            log_conv2d_output: Optional[bool] = False, log_grad_cam_heatmaps: Optional[bool] = False,
            log_target_conv_layer_kernels: Optional[bool] = False,
            grad_cam_heatmap_alpha_value: Optional[float] = 0.4, validation_batch_size: Optional[int] = None):
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
            epoch_frequency (int): The frequency (in epochs) at which Grad-CAM visuals should be generated and uploaded
              to WandB.
            validation_batch_size (int): The number of images in a validation dataset batch. If None, it is expected
              that the validation dataset has already been unbatched.
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
            seed (Optional[int]): The random seed to use for reproducibility.
        """
        self._num_images = num_images
        self._num_classes = num_classes
        self._wab_trial_run = wab_trial_run
        self._target_conv_layer_name = target_conv_layer_name
        self._validation_data = validation_data
        self._epoch_frequency = epoch_frequency
        self._validation_batch_size = validation_batch_size
        self._validation_steps = validation_steps
        self._log_conv2d_output = log_conv2d_output
        self._log_grad_cam_heatmaps = log_grad_cam_heatmaps
        self._log_target_conv_layer_kernels = log_target_conv_layer_kernels
        self._grad_cam_heatmap_alpha_value = grad_cam_heatmap_alpha_value

    def _grad_cam(self):
        logger.debug(f"Generating Grad-CAM visuals for {self._num_images} images on the validation dataset...")
        # Get image shape from the dataset (excluding the batch dimension):
        if self._validation_batch_size is not None:
            image_shape = tuple(self._validation_data.element_spec[0].shape[1:])
        else:
            image_shape = self._validation_data.element_spec[0].shape
        # Check to see if the provided validation dataset is infinite:
        if self._validation_data.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
            if self._validation_steps is None:
                raise ValueError(
                    "Since the provided validation dataset is infinite, the number of validation steps "
                    "must be specified.")
        if self._validation_batch_size is not None:
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
            plt.grid(False)
            plt.imshow(superimposed_img, cmap=jet_cm)
            plt.suptitle(f"Grad-CAM at Layer {self._target_conv_layer_name}")
            plt.colorbar()
            # plt.show()
            self._wab_trial_run.log({f'grad_cam_{self._target_conv_layer_name}': wab.Image(fig)})
            plt.clf()
            plt.close(fig)
        # Re-batch the validation dataset if necessary:
        if self._validation_batch_size is not None:
            self._validation_data = self._validation_data.batch(batch_size=self._validation_batch_size)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._epoch_frequency == 0:
            logger.debug(f"Logging Grad-CAM visuals at epoch {epoch} at a frequency of {self._epoch_frequency} epochs.")
            self._grad_cam()

    def on_train_end(self, logs=None):
        self._grad_cam()

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
                        if j + 1 > num_rows * num_cols:
                            break
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

import numpy as np
from loguru import logger
import seaborn as sns
from typing import Optional
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
            self, num_classes: int, wab_trial_run: Run, validation_data: Dataset,
            validation_steps: Optional[int] = None):
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
        super().__init__()

    def on_train_end(self, logs=None):
        """
        Called by the Keras framework at the end of Model training. This method is responsible for computing the
        Confusion Matrix on the validation dataset provided during callback initialization, and uploading the results
        to WaB.

        """
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
            axs[0, i].imshow(train_images[i])
            # Will definitely need to be changed for this to work with multi-class classification problems:
            axs[0, i].set_title('RG' if train_labels[i].numpy() == 1 else 'NRG')
            axs[0, i].axis('off')
            axs[1, i].imshow(val_images[i])
            axs[1, i].set_title('RG' if val_labels[i].numpy() == 1 else 'NRG')
            axs[1, i].axis('off')
        # Label the first row of images as training images and the second row as validation images:
        axs[0, 0].set_ylabel('Training Images')
        axs[1, 0].set_ylabel('Validation Images')
        plt.tight_layout()
        self._wab_trial_run.log({'train_val_images': wab.Image(fig)})

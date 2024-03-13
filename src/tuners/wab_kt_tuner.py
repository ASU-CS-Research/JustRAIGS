import os
import gc
from contextlib import redirect_stdout
import matplotlib.colors
import numpy as np
import keras_tuner as kt
from keras_tuner.engine.trial import Trial
import tensorflow as tf
import wandb as wab
from loguru import logger
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from wandb.apis.public import Run
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
# from typing_extensions import override
from typing import overload, Optional, List
import keras
import seaborn as sns
from IPython.display import Image, display

from src.piping_detection.metrics import BalancedBinaryAccuracy
# from metrics import BalancedBinaryAccuracy


class WBTuner(kt.Tuner):
    """
    Custom Weights & Biases Tuner subclassed from :class:`kt.Tuner`. Integrates Keras Tuner with Weights and Biases for
    versioning and result logging.

    See Also:
        https://wandb.ai/arig23498/keras-tuner/reports/Automate-Hyperparameter-Tuning-Using-Keras-Tuner-and-W-B--Vmlldzo0MzQ1NzU
    """

    def __init__(
            self, oracle: kt.Oracle, seed: int, wab_group_name: str, logging_frequency_in_epochs: Optional[int] = 100, **kwargs):
        super().__init__(oracle, **kwargs)
        # self._hyper_model = None
        self._wab_group_name = wab_group_name
        self._seed = seed
        self._logging_frequency_in_epochs = logging_frequency_in_epochs

    # @property
    # def hyper_model(self) -> Optional[kt.HyperModel]:
    #     return self._hyper_model

    @tf.function
    def run_train_step_on_batch(self, batched_image_data: tf.Tensor, batched_image_labels: tf.Tensor):
        """
        Called once per-batch during training. This method returns the loss for the present batch.

        .. todo:: Docstrings.

        See Also:
            https://wandb.ai/arig23498/keras-tuner/reports/Automate-Hyperparameter-Tuning-Using-Keras-Tuner-and-W-B--Vmlldzo0MzQ1NzU
        """
        with tf.GradientTape() as tape:
            logger.debug(f"type(hypermodel): {type(self.hypermodel)}")
            logits = self.hypermodel.model(hp=self.hypermodel.hyperparameters, x=batched_image_data, y=batched_image_labels)
            loss = self.hypermodel.model.loss(batched_image_labels, logits)
            gradients = tape.gradient(loss, self.hypermodel.model.trainable_variables)
            optimizer = self.hypermodel.model.optimizer
            optimizer.apply_gradients(zip(gradients, self.hypermodel.model.trainable_variables))
            self.hypermodel.model.loss_tracker.update_state(loss)
        return loss

    def generate_and_upload_confusion_matrix(
            self, trial_run: Run, y_true: np.array, y_pred: np.array, class_names: np.array,
            probs: Optional[np.ndarray] = None):
        logger.debug(f"Generating confusion matrix...")
        num_classes = 2
        confusion_matrix = tf.math.confusion_matrix(
            labels=y_true, predictions=y_pred, num_classes=num_classes, dtype=tf.int32
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
        trial_run.log({'confusion_matrix': wab.Image(fig)})
        plt.clf()
        plt.close(fig)

        # trial_run.log({
        #     'confusion_matrix': wab.plot.confusion_matrix(
        #         probs=probs, y_true=y_true, preds=y_pred, class_names=class_names
        #     )}
        # )

    def run_grad_cam_and_upload_results(
            self, wab_trial_run: Run, images: np.ndarray, labels: np.array, model: tf.keras.Model,
            last_conv_layer_name: str, heatmap_alpha: Optional[float] = 0.4, log_conv_output: Optional[bool] = False,
            log_heatmap: Optional[bool] = False, log_last_conv_layer_kernels: Optional[bool] = False):
        """
        Run Grad-CAM on x random images from each class:

        See Also: https://keras.io/examples/vision/grad_cam/
        """
        if log_last_conv_layer_kernels:
            # Visualize the kernels for the last convolutional layer.
            last_conv_layer = model.get_layer(last_conv_layer_name)
            num_kernels = last_conv_layer.kernel.shape[-1]
            fig = plt.figure(num=1)
            for i in range(num_kernels):
                kern = last_conv_layer.kernel.numpy()[..., i]
                plt.title(f"Kernel [{i+1}/{num_kernels}]")
                plt.imshow(kern, cmap='gray')
                plt.colorbar()
                plt.grid(None)
                wab_trial_run.log({f'last_conv_layer_kernels': wab.Image(fig)})
                # plt.show()
                plt.clf()

        def get_activation_heatmap_for_images(
                model: tf.keras.Model, images: np.ndarray, log_conv_output: Optional[bool] = False,
                log_heatmap: Optional[bool] = False) -> np.ndarray:
            last_conv_layer = model.get_layer(last_conv_layer_name)
            conv_layer_output_activation_shape = last_conv_layer.output_shape[1:-1]
            num_images = images.shape[0]
            heatmaps = np.zeros((num_images, *conv_layer_output_activation_shape))
            # New model to map the input image to the activations of the last conv layer as well as the output predictions:
            grad_cam_model = tf.keras.Model(model.inputs, [last_conv_layer.output, model.output], name='grad_cam_model')
            pred_index = None
            for i, image in enumerate(images):
                image = np.expand_dims(image, axis=0)
                # Compute the gradient of the top predicted class for the input image with respect to the activations of the
                # last conv layer:
                with tf.GradientTape() as tape:
                    last_conv_layer_output, preds = grad_cam_model(image)
                    # preds is now shape (5, 1) and last_conv_layer_output is now shape (5, 206, 206, 64)
                    if pred_index is None:
                        pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]
                # The gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the
                # last conv layer:
                grads = tape.gradient(class_channel, last_conv_layer_output)
                # Create a vector where each entry is the mean intensity of the gradient over a specific feature map
                # channel:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                # Multiply each channel in the feature map array by "how important this channel is" with regard to the top
                # predicted class. Then sum all channels to obtain the heatmap class activation:
                last_conv_layer_output = last_conv_layer_output[0]
                if log_conv_output:
                    num_channels = last_conv_layer_output.shape[-1]
                    if num_channels == 1:
                        fig = plt.figure(num=1)
                        plt.suptitle(f"Output of {last_conv_layer.name} for {model.name}")
                        plt.title(f"{num_channels} Channel")
                        plt.imshow(last_conv_layer_output, cmap='gray')
                        plt.grid(None)
                        plt.colorbar()
                        plt.xlabel('Pixel Index')
                        plt.ylabel('Pixel Index')
                        # plt.show()
                        wab_trial_run.log({f'{last_conv_layer.name}_output': wab.Image(fig)})
                        plt.clf()
                        plt.close(fig)
                    else:
                        fig = plt.figure(num=1)
                        plt.suptitle(f"Output of {last_conv_layer.name} for Model {model.name}")
                        plt.title(f"Mean Across {num_channels} Channels")
                        plt.imshow(tf.reduce_mean(last_conv_layer_output, axis=-1), cmap='gray')
                        plt.grid(None)
                        plt.colorbar()
                        plt.xlabel('Pixel Index')
                        plt.ylabel('Pixel Index')
                        # plt.show()
                        wab_trial_run.log({f'{last_conv_layer.name}_output': wab.Image(fig)})
                        plt.clf()
                        plt.close(fig)
                heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                # Normalize the heatmap between 0 and 1 for visualization purposes:
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                # Plot the image for debugging purposes:
                if log_heatmap:
                    plt.matshow(heatmap.numpy(), cmap='jet')
                    plt.title(f"Grad-CAM Weight Heatmap at {last_conv_layer.name} for Model {model.name}")
                    plt.colorbar()
                    plt.grid(None)
                    plt.xlabel('Pixel Index')
                    plt.ylabel('Pixel Index')
                    # plt.show()
                    fig = plt.gcf()
                    wab_trial_run.log({f'weight_heatmap_{last_conv_layer.name}': wab.Image(fig)})
                    plt.clf()
                    plt.close(fig)
                heatmaps[i] = heatmap.numpy()
            return heatmaps

        # Get the raw heatmap of the activations in the last convolutional layer:
        heatmaps = get_activation_heatmap_for_images(
            model=model, images=images, log_conv_output=log_conv_output, log_heatmap=log_heatmap
        )
        for i, image in enumerate(images):
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
            superimposed_img = jet_heatmap * heatmap_alpha + (image * 255)
            superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
            fig = plt.figure(num=1)
            plt.imshow(superimposed_img, cmap=jet_cm)
            plt.suptitle(f"Grad-CAM at Layer {last_conv_layer_name} for Model {model.name}")
            plt.xlabel("Pixel Index")
            plt.ylabel("Pixel Index")
            plt.colorbar()
            plt.grid(None)
            # plt.show()
            wab_trial_run.log({f'grad_cam_{last_conv_layer_name}': wab.Image(fig)})
            plt.clf()
            plt.close(fig)

    # @override
    def run_trial(self, trial: Trial, dataset, callbacks: tf.keras.callbacks, num_experiments_per_trial: Optional[int] = 1, *args, **kwargs):
        """
        Overrides the :meth:`kt.Tuner.run_trial` method.

        Args:
            trial (kt.Trial): A :class:`kt.Trial` instance.
            dataset (tf.data.Dataset): A :class:`tf.data.Dataset` instance (may be a :class:`PrefetchDataset` at
              runtime).
            num_experiments_per_trial (int): Number of times to completely retrain the model per-trial.
            callbacks (tf.keras.callbacks): A :class:`tf.keras.callbacks` instance containing callbacks which should be
              executed during training (such as early stopping).
            *args: Variable length argument list passed through from the call to :meth:`tuner.search` in ``hypermodel``.
            **kwargs: Arbitrary keyword arguments passed through from the call to :meth:`tuner.search` in ``hypermodel``.

        Returns:
            None
        """
        # WANDB Initialization for the trial:
        trial_run = wab.init(
            project="PipingDetection",
            config=trial.hyperparameters.values,
            group=self._wab_group_name
        )
        # Build the model with the set hyperparameters for this particular trial:
        model = self.hypermodel.build(hp=trial.hyperparameters, weights='imagenet')
        # Log the model summary to weights and biases:
        trial_run.log({"model_summary": model.summary()})
        # Log the model summary to a text file and upload it as an artifact to weights and biases:
        with open("model_summary.txt", "w") as fp:
            with redirect_stdout(fp):
                model.summary()
        model_summary_artifact = wab.Artifact("model_summary", type='model_summary')
        model_summary_artifact.add_file("model_summary.txt")
        trial_run.log_artifact(model_summary_artifact)
        # for execution_index in range(self.executions_per_trial):
        for experiment_index in range(num_experiments_per_trial):
            logger.debug(f"Calling .fit() on model: {model}")
            history = self.hypermodel.fit(
                model=model, weights_and_biases_trial_run_object=trial_run, x=dataset, *args, **kwargs
            )
            # model.fit(x=dataset, weights_and_biases_trial_run_object=trial_run, callbacks=callbacks, *args, **kwargs)
        # Save the model:
        models_dir = os.path.abspath('models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, f"{trial.trial_id}.h5")
        if os.path.exists(model_path):
            os.remove(model_path)
        model.save(model_path)
        # Check that the model was saved correctly:
        # Load the model with the custom metric (see https://www.tensorflow.org/guide/keras/serialization_and_saving#custom_objects):
        # custom_objects = {'BalancedBinaryAccuracy': BalancedBinaryAccuracy}
        # with tf.keras.utils.custom_object_scope(custom_objects):
        #     reconstructed_model = keras.models.load_model(os.path.join(models_dir, f"{trial.trial_id}.h5"))
        # test_input = dataset.take(1)
        # test_image, test_label = test_input.get_single_element()
        # np.testing.assert_allclose(
        #     model.predict(test_image), reconstructed_model.predict(test_image)
        # )
        # del reconstructed_model
        # Save the history:
        histories_dir = os.path.abspath('histories')
        if not os.path.exists(histories_dir):
            os.makedirs(histories_dir)
        np.save(os.path.join(histories_dir, f"history_{trial.trial_id}.npy"), history.history)
        # Save the trial:
        trials_dir = os.path.abspath('trials')
        if not os.path.exists(trials_dir):
            os.makedirs(trials_dir)
        trial.save(os.path.join(trials_dir, f"trial_{trial.trial_id}.json"))
        # Upload the saved model to Weights and Biases:
        trial_run.save(model_path)
        # Get image size from dataset:
        image_shape = tuple(dataset.element_spec[0].shape[1:])
        # Check to see if the provided training dataset is infinite:
        if dataset.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
            # Get the number of steps per epoch:
            try:
                steps_per_epoch = kwargs['steps_per_epoch']
            except KeyError as err:
                raise ValueError("If the provided dataset is infinite, the number of steps per-epoch must be "
                                 "specified.")
        else:
            steps_per_epoch = None
        # Check to see if a validation dataset was provided:
        try:
            validation_data = kwargs['validation_data']
        except KeyError as err:
            validation_data = None
            validation_steps = None
        if validation_data is not None:
            # Check to see if the provided validation dataset is infinite:
            if validation_data.cardinality().numpy() == tf.data.INFINITE_CARDINALITY:
                try:
                    validation_steps = kwargs['validation_steps']
                except KeyError as err:
                    raise ValueError("If the provided validation dataset is infinite, the number of validation steps "
                                     "must be specified.")
            else:
                validation_steps = None
        else:
            validation_steps = None
        # Plot confusion matrix:
        y_pred = np.array([])
        y_true = np.array([])
        class_names = np.array([])
        for i, batch in enumerate(dataset):
            if steps_per_epoch is not None:
                if i == steps_per_epoch:
                    break
            images = batch[0]
            labels = batch[1]
            y_true = np.concatenate((y_true, labels.numpy()))
            y_pred_batch = model.predict_on_batch(images).squeeze()
            class_names_batch = np.array(['Piping' if label.numpy() == 1 else 'NoPiping' for label in labels])
            y_pred = np.concatenate((y_pred, y_pred_batch))
            class_names = np.concatenate((class_names, class_names_batch))
        y_pred = np.array([0 if pred <= 0.5 else 1 for pred in y_pred])
        y_true = y_true.squeeze()
        class_names = class_names.squeeze()
        self.generate_and_upload_confusion_matrix(
            trial_run=trial_run, y_true=y_true, y_pred=y_pred, class_names=class_names, probs=None
        )
        # Gradient activation maps:
        num_images_to_map = 5
        dataset = dataset.unbatch()
        random_batch = dataset.take(num_images_to_map)
        random_images = np.zeros((num_images_to_map, *image_shape))
        random_labels = np.zeros((num_images_to_map, 1))
        for i, (image, label) in enumerate(random_batch):
            random_images[i] = image.numpy()
            random_labels[i] = label.numpy()
        # .. todo:: Dynamically retrieve the name of the last convolutional layer from the model, or pass it through.
        self.run_grad_cam_and_upload_results(
            wab_trial_run=trial_run, model=model, images=random_images, labels=random_labels,
            last_conv_layer_name='conv2d_1', log_conv_output=True, log_heatmap=True, log_last_conv_layer_kernels=True
        )
        # Finish the WANDB run:
        trial_run.finish()
        tf.keras.backend.clear_session()
        return history

import os
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from src.models.models import InceptionV3WaBModel, WaBModel
from src.utils.datasets import load_datasets
from src.callbacks.custom import GradCAMCallback
from loguru import logger
from matplotlib import cm


def gradCAM(model_dir: str, image_batch: tf.Tensor, layer_name: str, num_classes: Optional[int] = 1, inner: Optional[str] = None,
            alpha = 0.5):
    """

    Args:
        model_dir (str): path to the directory containing the saved model.
        image_batch (tf.Tensor): a batch of images to visualize the GradCAM heatmap for.
        layer_name (str): name of the layer to visualize.
        num_classes (Optional[int]): number of classes the model was trained on. Defaults to 1, in which case the model
            is assumed to be a binary classifier.
        inner (Optional[str]): name of layer to find ``layer_name`` in. Defaults to None, in which case the model is
            searched for the layer with the name ``layer_name``.

    Returns:
        plt.Figure: a matplotlib figure showing the GradCAM heatmap for the input images.
    """
    # Load the model:
    model = tf.keras.models.load_model(model_dir, compile=False)
    images = image_batch.numpy()
    if inner is not None:
        target_conv_layer = model.get_layer(inner).get_layer(layer_name)
    else:
        target_conv_layer = model.get_layer(layer_name)
    conv_layer_output_activation_shape = target_conv_layer.output_shape[1:-1]
    num_images = images.shape[0]
    heatmaps = np.zeros((num_images, *conv_layer_output_activation_shape))
    # New model to map the input image to the activations of the last conv layer as well as the output predictions:
    inputs = model.inputs if inner is None else model.get_layer(inner).inputs
    outputs = model.outputs if inner is None else model.get_layer(inner).outputs
    grad_cam_model = tf.keras.Model(
        inputs, [target_conv_layer.output, outputs], name='grad_cam_model'
    )
    pred_index = None
    for i, image in enumerate(images):
        image = np.expand_dims(image, axis=0)
        # Compute the gradient of the top predicted class for the input image with respect to the activations of the
        # last conv layer:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_cam_model(image.squeeze(0), training=False)
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
        plt.clf()
        # Multiply each channel in the feature map array by "how important this channel is" with regard to the top
        # predicted class. Then sum all channels to obtain the heatmap class activation:
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        # Normalize the heatmap between 0 and 1 for visualization purposes:
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        # Plot the image for debugging purposes:
        plt.matshow(heatmap.numpy(), cmap='jet')
        plt.title(f"{target_conv_layer.name} Grad-CAM Heatmap")
        plt.colorbar()
        plt.grid(None)
        # plt.show()
        fig = plt.gcf()
        plt.clf()
        plt.close(fig)
        # heatmaps[i] = heatmap.numpy()

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
        superimposed_img = (jet_heatmap * alpha) + ((image * 255) * (1 - alpha))
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        fig = plt.figure(num=1)
        plt.grid(False)
        plt.imshow(superimposed_img, cmap=jet_cm)
        plt.suptitle(f"Grad-CAM at Layer {target_conv_layer}")
        plt.colorbar()
        plt.show()
        plt.clf()
        plt.close(fig)

    return heatmaps


"""
Example code showing how to load in a previously trained model and perform inference. 
"""

if __name__ == '__main__':
    # Declare constants:
    SEED = 42
    BATCH_SIZE = 10
    NUM_IMAGES_TO_SHOW = 5  # Number of images to visualize after performing inference.
    # Set the seed for reproducibility:
    tf.random.set_seed(SEED)
    # Path to the root directory of the repository on the lambda machine:
    repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    # # Create a `models` folder under JustRAIGS and store the model you wish to use for inference there:
    # models_folder_path = os.path.join(repo_root_dir, 'models')
    # if not os.path.exists(models_folder_path):
    #     os.makedirs(models_folder_path)
    # # Path to the model you wish to use for inference:
    # best_model_path = os.path.join(models_folder_path, 'model-best.h5')
    # # Specify the loss function, optimizer, and metrics to use when evaluating the model:
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # metrics = ['binary_accuracy']
    # '''
    # Define custom objects for loading the model (see:
    # https://www.tensorflow.org/guide/keras/serialization_and_saving#passing_custom_objects_to_load_model):
    # '''
    # custom_objects = {
    #     'WaBModel': WaBModel,
    #     'InceptionV3WaBModel': InceptionV3WaBModel,
    # }
    # with tf.keras.utils.custom_object_scope(custom_objects):
    #     # Load the model:
    #     model = tf.keras.models.load_model(best_model_path, compile=False)
    # # Compile the model:
    # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # # Check the summary:
    # model.summary()
    saved_models = os.path.join(repo_root_dir, 'saved_models')
    binary_model_dir = os.path.join(saved_models, 'binary')
    multi_model_dir = os.path.join(saved_models, 'multi')
    binary_model = tf.saved_model.load(binary_model_dir)
    logger.debug(f"Loaded binary model from: {binary_model_dir}")
    multi_model = tf.saved_model.load(multi_model_dir)
    logger.debug(f"Loaded multi-label model from: {multi_model_dir}")
    binary_inference = binary_model.signatures['serving_default']
    multi_inference = multi_model.signatures['serving_default']

    # Defining inference functions for ease of access:
    def binary_inference_fn(image_batch):
        return binary_inference(image_batch)['dense_1']
    def multi_inference_fn(image_batch):
        return multi_inference(image_batch)['dense']

    logger.debug("Loading in datasets for inference...")
    # Load datasets to perform inference on:
    train_ds, val_ds, test_ds = load_datasets(
        color_mode='rgb', target_size=(75, 75), interpolation='bilinear', keep_aspect_ratio=False,
        train_set_size=0.6, val_set_size=0.2, test_set_size=0.2, seed=SEED, num_partitions=6, batch_size=BATCH_SIZE,
        num_images=None, oversample_train_set=True, oversample_val_set=True
    )
    logger.debug("Datasets loaded successfully.")
    # Use the trained model to issue predictions on the validation set (images it did not see during training):
    num_correct_predictions = 0
    image_shape_no_batch_dim = tuple(val_ds.element_spec[0].shape[1:])
    images = np.zeros((NUM_IMAGES_TO_SHOW, *image_shape_no_batch_dim), dtype=np.uint8)
    true_labels = np.zeros((NUM_IMAGES_TO_SHOW,), dtype=np.uint8)
    pred_labels = np.zeros((NUM_IMAGES_TO_SHOW,), dtype=np.uint8)
    correctly_predicted_random_images = np.zeros((NUM_IMAGES_TO_SHOW, *image_shape_no_batch_dim))
    correctly_predicted_random_labels = np.zeros((NUM_IMAGES_TO_SHOW, 1))
    correctly_predicted_random_probs = np.zeros((NUM_IMAGES_TO_SHOW, 1))
    # Un-batch the validation dataset (no need for mini-batches when performing inference):
    unbatched_val_ds = val_ds.unbatch()
    # Iterate over NUM_IMAGES_TO_SHOW examples and issue predictions:
    for i, (image, label) in enumerate(unbatched_val_ds):
        if num_correct_predictions == NUM_IMAGES_TO_SHOW:
            break
        # Model expects input to be (None, width, height, num_channels) so prepend a batch dimension:
        image = tf.expand_dims(image, axis=0)
        images[i] = image.numpy().astype(np.uint8)
        true_labels[i] = label.numpy()
        # Run inference:
        pred_prob = binary_inference_fn(image)
        y_pred_prob = tf.squeeze(binary_inference_fn(image), axis=-1)
        # This assumes binary classification (of course), so modify if not the case:
        y_pred = 0 if y_pred_prob <= 0.5 else 1
        # if y_pred == 1:
        #     multi_classification = multi_inference_fn(image)
        pred_labels[i] = y_pred
        if y_pred == label.numpy():
            correctly_predicted_random_images[num_correct_predictions] = image.numpy()
            correctly_predicted_random_labels[num_correct_predictions] = label.numpy()
            correctly_predicted_random_probs[num_correct_predictions] = pred_prob
            num_correct_predictions += 1
    '''
    Plot the images and their corresponding labels:
    '''
    fig, axes = plt.subplots(nrows=1, ncols=NUM_IMAGES_TO_SHOW, figsize=(20, 20))
    plt.suptitle("Validation Set Images and Predicted Labels")
    plt.grid(False)
    # sort random images and labels by probs:
    sort_indices = np.argsort(correctly_predicted_random_probs, axis=0)
    correctly_predicted_random_images = correctly_predicted_random_images[sort_indices]
    correctly_predicted_random_labels = correctly_predicted_random_labels[sort_indices]
    correctly_predicted_random_probs = correctly_predicted_random_probs[sort_indices]

    for i in range(NUM_IMAGES_TO_SHOW):
        axis = axes[-i]
        axis.imshow(images[-i])
        axis.set_title(f"Y_true: {true_labels[-i]}, Y_pred: {pred_labels[-i]}")
    # # adjust spacing between subplots:
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Next, an example of running GradCAM on the correctly predicted images:
    # Specify the layer to visualize:
    # Binary classifier is EfficientNetB7, so the last conv layer is 'top_activation':
    gradCAM(
        binary_model_dir, tf.convert_to_tensor(correctly_predicted_random_images), 'top_activation',
        num_classes=1, inner='efficientnetb7'
    )


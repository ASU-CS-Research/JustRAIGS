import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.models.models import InceptionV3WaBModel, WaBModel
from src.utils.datasets import load_datasets

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
    # Create a `models` folder under JustRAIGS and store the model you wish to use for inference there:
    models_folder_path = os.path.join(repo_root_dir, 'models')
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
    # Path to the model you wish to use for inference:
    best_model_path = os.path.join(models_folder_path, 'model-best.h5')
    # Specify the loss function, optimizer, and metrics to use when evaluating the model:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    metrics = ['binary_accuracy']
    '''
    Define custom objects for loading the model (see: 
    https://www.tensorflow.org/guide/keras/serialization_and_saving#passing_custom_objects_to_load_model):
    '''
    custom_objects = {
        'WaBModel': WaBModel,
        'InceptionV3WaBModel': InceptionV3WaBModel,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Load the model:
        model = tf.keras.models.load_model(best_model_path, compile=False)
    # Compile the model:
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # Check the summary:
    model.summary()
    # Load datasets to perform inference on:
    train_ds, val_ds, test_ds = load_datasets(
        color_mode='rgb', target_size=(75, 75), interpolation='bilinear', keep_aspect_ratio=False,
        train_set_size=0.6, val_set_size=0.2, test_set_size=0.2, seed=SEED, num_partitions=6, batch_size=BATCH_SIZE,
        num_images=50, oversample_train_set=True, oversample_val_set=True
    )
    # Use the trained model to issue predictions on the validation set (images it did not see during training):
    num_correct_predictions = 0
    image_shape_no_batch_dim = tuple(val_ds.element_spec[0].shape[1:])
    images = np.zeros((NUM_IMAGES_TO_SHOW, *image_shape_no_batch_dim), dtype=np.uint8)
    true_labels = np.zeros((NUM_IMAGES_TO_SHOW,), dtype=np.uint8)
    pred_labels = np.zeros((NUM_IMAGES_TO_SHOW,), dtype=np.uint8)
    correctly_predicted_random_images = np.zeros((NUM_IMAGES_TO_SHOW, *image_shape_no_batch_dim))
    correctly_predicted_random_labels = np.zeros((NUM_IMAGES_TO_SHOW, 1))
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
        pred_prob = model(image, training=False)
        y_pred_prob = tf.squeeze(model(image, training=False), axis=-1)
        # This assumes binary classification (of course), so modify if not the case:
        y_pred = 0 if y_pred_prob <= 0.5 else 1
        pred_labels[i] = y_pred
        if y_pred == label.numpy():
            correctly_predicted_random_images[num_correct_predictions] = image.numpy()
            correctly_predicted_random_labels[num_correct_predictions] = label.numpy()
            num_correct_predictions += 1
    '''
    Plot the images and their corresponding labels:
    '''
    fig, axes = plt.subplots(nrows=1, ncols=NUM_IMAGES_TO_SHOW, figsize=(20, 20))
    plt.suptitle("Validation Set Images and Predicted Labels")
    plt.grid(False)
    for i in range(NUM_IMAGES_TO_SHOW):
        axis = axes[i]
        axis.imshow(images[i])
        axis.set_title(f"Y_true: {true_labels[i]}, Y_pred: {pred_labels[i]}")
    plt.tight_layout()
    plt.show()
    plt.clf()

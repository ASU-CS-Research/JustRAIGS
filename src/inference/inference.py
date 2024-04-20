import numpy
from PIL import Image
from src.inference.helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks,load_and_preprocess_image, save_predictions, append_prediction 
import os
import random
import tensorflow as tf
import pandas as pd


def run():
    _show_torch_cuda_info()

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        image = Image.open(jpg_image_file_name)
        numpy_array = numpy.array(image)

        is_referable_glaucoma_likelihood = random.random()
        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5
        if is_referable_glaucoma:
            features = {
                k: random.choice([True, False])
                for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()
            }
        else:
            features = None
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def run_inference_tasks(model_path: str, image_preprocessing_fn: callable):
    input_size_no_batch = (75, 75, 3)
    _show_tf_cuda_info()
    if not os.path.isdir(model_path):
        error_message = f"Model path: {model_path} is not a directory or does not have subfolders binary or multi. Expecting a SavedModel format directory."
        print(error_message)
        raise ValueError(error_message)
    # Need to import binary and muli-label
    bin_model = tf.keras.models.load_model(os.path.join(model_path, "binary"))
    multi_model = tf.keras.models.load_model(os.path.join(model_path, "multi"))
    image_filename_and_callback_df = pd.DataFrame(inference_tasks(), columns=["image_filename", "callback"])
    image_series = image_filename_and_callback_df['image_filename'].apply(image_preprocessing_fn)
    print(image_series)
    # Only one callback function
    #Convert to tf Dataset
    predict_ds = tf.data.Dataset.from_tensor_slices(list(image_series)).batch(10)
    # Predict
    binary_pred = bin_model.predict(predict_ds)
    binary_pred = list(map(float, binary_pred))
    multi_pred = multi_model.predict(predict_ds)
    for i, res in enumerate(binary_pred):
        is_RG = res > 0.5
        likelyhood_RG = res
        features = None
        if is_RG:
            feature_values = list([bool(p > 0.5) for p in multi_pred[i]]) # converts np bool to python bool typr. bool_ not JSON compatable
            features = dict(zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), feature_values))
        # Save results
        append_prediction(is_RG, res, features)
    save_predictions()

def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

def _show_tf_cuda_info():
    print("=+=" * 10)
    print(f"TF CUDA is available: {(available := tf.test.is_built_with_cuda())}")
    if available:
        print(f"TF CUDA is available: {(available := tf.config.list_physical_devices('GPU'))}")
    print("=+=" * 10)


if __name__ == "__main__":
    MODEL_PATH = os.path.abspath("saved_models")
    IMAGE_PREPROCESSING_FN = load_and_preprocess_image
    raise SystemExit(run_inference_tasks(MODEL_PATH, IMAGE_PREPROCESSING_FN))

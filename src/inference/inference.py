import numpy
from PIL import Image
from src.inference.helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import os
import random
import tensorflow as tf
from src.utils.datasets import load_and_preprocess_image
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
    bin_model = tf.saved_model.load(model_path + "/binary")
    multi_model = tf.saved_model.load(model_path + "/multi")
    binary_inference = bin_model.signatures["serving_default"] # output is a dictionary, use key: "dense_1" for binary
    # Make sure this model is the correct one, it should output a single value
    random_image = tf.random.uniform((1, *input_size_no_batch))
    output = binary_inference(random_image)["dense_1"]
    print(f"Loaded in binary inference model, got output shape: {output.numpy().shape} with value {output.numpy()}")
    multi_inference = multi_model.signatures["serving_default"] # output is a dictionary, use key: "dense" for multi
    # Make sure this model is the correct one, it should output ten values
    output = multi_inference(random_image)["dense"]
    print(f"Loaded in multi inference model, got output shape: {output.numpy().shape} with value {output.numpy()}")
    image_filename_and_callback_df = pd.DataFrame(inference_tasks(), columns=["image_filename", "callback"])
    image_filename_and_callback_df.map(image_preprocessing_fn)
    adr_set = set(image_filename_and_callback_df.map(id))
    for addr in adr_set:
        print(addr)
    #Convert to tf Dataset
    predict_ds = tf.data.Dataset.from_tensor_slices(list(image_filename_and_callback_df["image_filename"]), list(image_filename_and_callback_df["callback"]))
    # Predict

    # Save results


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

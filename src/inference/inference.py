import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import os
import random
import tensorflow as tf
from src.utils.datasets import load_and_preprocess_image


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
    _show_tf_cuda_info()
    if not os.path.isdir(model_path):
        error_message = f"Model path: {model_path} is not a directory. Expecting a SavedModel format directory."
        print(error_message)
        raise ValueError(error_message)
    model = tf.saved_model.load(model_path)
    inference = model.signatures["serving_default"]
    for jpg_image_file_name, save_prediction in inference_tasks():




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
        print(f"\tTF CUDA is available: {(available := tf.test.is_gpu_available())}")
        print(f"\tTF CUDA is available: {(available := tf.config.list_physical_devices('GPU'))}")
    print("=+=" * 10)


if __name__ == "__main__":
    MODEL_PATH = os.path.abspath("")
    IMAGE_PREPROCESSING_FN = load_and_preprocess_image
    raise SystemExit(run_inference_tasks(MODEL_PATH, IMAGE_PREPROCESSING_FN))
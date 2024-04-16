import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import os
import random
import tensorflow as tf


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


def run_inference_tasks(model_path: str):
    if not os.path.isdir(model_path):
        error_message = f"Model path: {model_path} is not a directory. Expecting a SavedModel format directory."
        print(error_message)
        raise ValueError(error_message)
    model = tf.saved_model.load(model_path)
    inference = model.signatures["serving_default"]



def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    MODEL_PATH = os.path.abspath("")
    raise SystemExit(run(MODEL_PATH))
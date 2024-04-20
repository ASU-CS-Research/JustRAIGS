import json
import os
import tempfile
from pathlib import Path
import shutil
from pprint import pprint
import numpy as np
from typing import Tuple, Union
import SimpleITK as sitk
from PIL import Image
import keras
from loguru import logger

REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))
logger.debug(f"REPO_ROOT_DIR: {REPO_ROOT_DIR}")

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}

INPUT_DIR = os.path.abspath(os.path.join(REPO_ROOT_DIR, "test/input")) # Change these back to /input and /output before turning it in
OUTPUT_DIR = os.path.abspath(os.path.join(REPO_ROOT_DIR, "test/output"))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
logger.debug(f"INPUT_DIR: {INPUT_DIR}")
logger.debug(f"OUTPUT_DIR: {OUTPUT_DIR}")
TEMP_DIR = os.path.abspath("tmp")

is_referable_glaucoma_stacked = []
is_referable_glaucoma_likelihood_stacked = []
glaucomatous_features_stacked = []

def inference_tasks():
    input_files = [x for x in Path(INPUT_DIR).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)

    for file_path in input_files:
        if file_path.suffix == ".JPG":  # A single image
            yield file_path, append_prediction #single_file_inference(image_file=file_path, callback=save_prediction)
        elif file_path.suffix == ".mba":
            yield from single_file_inference(file_path, append_prediction)
        elif file_path.suffix == ".tiff":  # A stack of images
            yield from stack_inference(stack=file_path, callback=append_prediction) # need to modify this line

    #write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    #write_referable_glaucoma_decision_likelihood(
    #    is_referable_glaucoma_likelihood_stacked
    #)
    #write_glaucomatous_features(glaucomatous_features_stacked)

# Moved from inference_tasks method
def append_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
    ):
        is_referable_glaucoma_stacked.append(is_referable_glaucoma)
        is_referable_glaucoma_likelihood_stacked.append(likelihood_referable_glaucoma)
        if glaucomatous_features is not None:
            glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)

#moved from inference_tasks method
def save_predictions():
    write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    write_referable_glaucoma_decision_likelihood(
        is_referable_glaucoma_likelihood_stacked
    )
    write_glaucomatous_features(glaucomatous_features_stacked)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def single_file_inference(image_file, callback):
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    image = sitk.ReadImage(image_file)

    # Define the output file path
    output_path = Path(TEMP_DIR) / "image.jpg"

    # Save the 2D slice as a JPG file
    sitk.WriteImage(image, str(output_path))

    # Call back that saves the result
    def save_prediction(
        is_referable_glaucoma,
        likelihood_referable_glaucoma,
        glaucomatous_features=None,
    ):
        glaucomatous_features = (
            glaucomatous_features or DEFAULT_GLAUCOMATOUS_FEATURES
        )
        write_referable_glaucoma_decision([is_referable_glaucoma])
        write_referable_glaucoma_decision_likelihood(
            [likelihood_referable_glaucoma]
        )
        write_glaucomatous_features(
            [{**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features}]
        )

    yield output_path, callback


def stack_inference(stack, callback):
    de_stacked_images = []

    # Unpack the stack
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    #with tempfile.TemporaryDirectory() as temp_dir:
    with Image.open(stack) as tiff_image:

        # Iterate through all pages
        for page_num in range(tiff_image.n_frames):
            # Select the current page
            tiff_image.seek(page_num)

            # Define the output file path
            output_path = Path(TEMP_DIR) / f"image_{page_num + 1}.jpg"
            tiff_image.save(output_path, "JPEG")

            de_stacked_images.append(output_path)

            print(f"De-Stacked {output_path}")

        # Loop over the images, and generate the actual tasks
    for index, image in enumerate(de_stacked_images):
        # Call back that saves the result
        yield image, callback


def write_referable_glaucoma_decision(result):
    with open(os.path.join(OUTPUT_DIR,"multiple-referable-glaucoma-binary.json"), "w") as f:
        f.write(json.dumps(result))


def write_referable_glaucoma_decision_likelihood(result):
    with open(os.path.join(OUTPUT_DIR,"multiple-referable-glaucoma-likelihoods.json"), "w") as f:
        f.write(json.dumps(result))


def write_glaucomatous_features(result):
    with open(os.path.join(OUTPUT_DIR, "stacked-referable-glaucomatous-features.json"), "w") as f:
        f.write(json.dumps(result))


def load_and_preprocess_image(*args, **kwargs) -> Union[np.ndarray, float]:
    """
    Loads an image into a :class:`~numpy.ndarray` preprocess it, return the label as an :class:`int`.

    .. todo:: Add preprocessing logic here. Convert to grayscale, downsample, etc.

    Args:
        *args: Variable length argument list. The inputs depend on the :class:`~pandas.DataFrame` object (or
          :class:`~pandas.Series`) that the :meth:`~pandas.DataFrame.apply` method is called on.
        **kwargs: Arbitrary keyword arguments. The inputs depend on the :class:`~pandas.DataFrame` object (or
          :class:`~pandas.Series`) that the :meth:`~pandas.DataFrame.apply` method is called on.

    Within Args:
        color_mode:
        target_size:
        interpolation:
        keep_aspect_ratio:

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the image and its corresponding label.

    See Also:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply
        - https://keras.io/api/data_loading/image/#loadimg-function

    """
    image_abs_path = args[0]

    # logger.debug(f"load_img: {image_abs_path}")
    try:
        image = keras.utils.load_img(
            image_abs_path, color_mode='rgb', target_size=(75, 75, 3), interpolation='bilinear',
            keep_aspect_ratio=False
        )
    except Exception as e:
        print(f"Failed to load image: {image_abs_path}")
        print(f"Error: {e}")
        return float('nan')
    image = keras.utils.img_to_array(image)
    return image

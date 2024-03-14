import os
from pathlib import Path
from typing import Tuple, Optional
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
import pandas as pd
from loguru import logger


def load_and_preprocess_image(*args, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Loads an image into a :class:`~numpy.ndarray` preprocess it, return the label as an :class:`int`.

    .. todo:: Add preprocessing logic here. Convert to grayscale, downsample, etc.

    Args:
        *args: Variable length argument list. The inputs depend on the :class:`~pandas.DataFrame` object (or
          :class:`~pandas.Series`) that the :meth:`~pandas.DataFrame.apply` method is called on.
        **kwargs: Arbitrary keyword arguments. The inputs depend on the :class:`~pandas.DataFrame` object (or
          :class:`~pandas.Series`) that the :meth:`~pandas.DataFrame.apply` method is called on.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the image and its corresponding label.

    See Also:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply
        - https://keras.io/api/data_loading/image/#loadimg-function

    """
    image_abs_path = args[0][0]
    image_int_label = args[0][1]
    color_mode = args[1][1]
    target_size = args[2][1]
    interpolation = args[3][1]
    keep_aspect_ratio = args[4][1]
    image = keras.utils.load_img(
        image_abs_path, color_mode=color_mode, target_size=target_size, interpolation=interpolation,
        keep_aspect_ratio=keep_aspect_ratio
    )
    image = keras.utils.img_to_array(image)
    return image, image_int_label


def load_datasets(
        color_mode: str, target_size: Optional[Tuple[int, int]], interpolation: str, keep_aspect_ratio: bool,
        num_partitions: int, batch_size: int, num_images: Optional[int] = None, train_set_size: Optional[float] = 0.6,
        val_set_size: Optional[float] = 0.2, test_set_size: Optional[float] = 0.2, seed: Optional[int] = 42) -> [Dataset, Dataset, Dataset]:
    """
    Constructs TensorFlow train, val, test :class:`~tensorflow.data.Dataset` s from the provided high resolution image
    training set.

    Args:
        color_mode (str): One of "grayscale", "rgb", "rgba". Default: "rgb". The desired image format.
        target_size (Optional[Tuple[int, int]]): Either ``None`` (default to original size) or tuple of ints
          ``(img_height, img_width)``.
        interpolation (str): Interpolation method used to resample the image if the target size is different from that
          of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer
          is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are
          also supported. By default, "nearest" is used.
        keep_aspect_ratio (bool): Boolean, whether to resize images to a target size without aspect ratio distortion.
          The image is cropped in the center with target aspect ratio before resizing.
        num_partitions (int): An integer value representing the number of partitioned training sets to load in. The
          maximum value is ``6`` (Train_0 - Train_5), the minimum value is ``1`` (Train_0 only).
        batch_size (int): The batch size to use for the training, validation, and testing datasets.
        num_images (Optional[int]): The exact number of images to load from the training set. This value takes
          precedence over the specified number of partitions. However, if both ``num_partitions`` and ``num_images`` are
          specified then at most ``num_images`` will be loaded from the subset of partitions specified by
          ``num_partitions``, in other words the other partitions beyond the provided number will be untouched for
          data loading. This value is mostly useful for debugging downstream pipelines, and it defaults to :class:`None`
          indicating that all images in the partitions specified by ``num_partitions`` will be loaded.
        train_set_size (Optional[float]): The percentage of the data to use as the training dataset. This value defaults
          to ``0.6`` (e.g. ``60%``).
        val_set_size (Optional[float]): The percentage of the data to use as the validation dataset. This value defaults
          to ``0.2`` (e.g. ``20%``).
        test_set_size (Optional[float]): The percentage of the data to use as the testing dataset. This value defaults
          to ``0.2`` (e.g. ``20%``).
        seed (Optional[int]): The random seed used to shuffle the data and split the dataset into training, validation,
          and testing sets. This value defaults to ``42``.

    Returns:
        Tuple[:class:`~tensorflow.data.Dataset`, :class:`~tensorflow.data.Dataset`, :class:`~tensorflow.data.Dataset`]:
          A tuple containing the training, validation, and testing datasets.

    """
    high_res_train_data_path = os.path.abspath('/usr/local/data/JustRAIGS/raw/train')
    train_data_labels_csv_path = os.path.abspath(os.path.join(high_res_train_data_path, 'JustRAIGS_Train_labels.csv'))
    assert os.path.isfile(train_data_labels_csv_path), f"Failed to find {train_data_labels_csv_path}"
    train_data_labels_df = pd.read_csv(filepath_or_buffer=train_data_labels_csv_path, delimiter=';')
    train_image_abs_file_paths = []
    image_paths_df = pd.DataFrame()
    assert num_partitions <= 6, f"num_partitions must be less than or equal to 6, got {num_partitions}"
    assert num_partitions > 0, f"num_partitions must be greater than 0, got {num_partitions}"
    for i in range(num_partitions):
        train_set_partition_abs_path = os.path.join(high_res_train_data_path, f"{i}")
        for root, dirs, files in os.walk(train_set_partition_abs_path):
            for file_id, file in enumerate(files):
                train_image_abs_file_path = os.path.join(root, file)
                train_image_abs_file_paths.append(os.path.join(root, file))
                image_paths_df = image_paths_df.append(
                    {
                        'AbsPath': train_image_abs_file_path,
                        'Eye ID': file.split('.')[0]
                    }, ignore_index=True
                )
                if num_images is not None:
                    if file_id == num_images:
                        break
    # Modify the DataFrame to contain the absolute path to the 'Eye ID' training image:
    train_data_labels_df = train_data_labels_df.merge(image_paths_df, on=['Eye ID'], how='outer')
    del image_paths_df
    # Convert 'NRG' = 0 and 'RG' = 1
    final_label_int_mask = train_data_labels_df['Final Label'] == 'NRG'
    train_data_labels_df['Final Label Int'] = np.where(final_label_int_mask, 0, 1)
    # Drop rows with NaN absolute file paths:
    train_data_labels_df = train_data_labels_df[train_data_labels_df['AbsPath'].notna()]
    # Train, Validation, and Testing set partitioning:
    train_ds_df, val_ds_df = train_test_split(
        train_data_labels_df, train_size=train_set_size, test_size=val_set_size + test_set_size,
        shuffle=True, random_state=seed
    )
    logger.debug(f"train_ds_df.shape: {train_ds_df.shape}")
    val_ds_df, test_ds_df = train_test_split(
        val_ds_df, train_size=val_set_size, test_size=test_set_size, shuffle=False
    )
    logger.debug(f"val_ds_df.shape: {val_ds_df.shape}")
    logger.debug(f"test_ds_df.shape: {test_ds_df.shape}")
    '''
    At this point we have the dataframes partitioned into train, validation, and testing sets.
    Now we need to convert them to TensorFlow Datasets:
    '''
    train_img_and_labels_df = train_ds_df[['AbsPath', 'Final Label Int']].apply(
        load_and_preprocess_image, axis=1, raw=True, result_type='reduce', args=(
            ('color_mode', color_mode), ('target_size', target_size), ('interpolation', interpolation),
            ('keep_aspect_ratio', keep_aspect_ratio)
        )
    )
    train_img_and_labels_df.rename(columns={'AbsPath': 'NpImage', 'Final Label Int': 'LabelTensor'}, inplace=True)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_img_and_labels_df['NpImage']), list(train_img_and_labels_df['LabelTensor']))
    )
    del train_ds_df
    del train_img_and_labels_df
    val_img_and_labels_df = val_ds_df[['AbsPath', 'Final Label Int']].apply(
        load_and_preprocess_image, axis=1, raw=True, result_type='reduce', args=(
            ('color_mode', color_mode), ('target_size', target_size), ('interpolation', interpolation),
            ('keep_aspect_ratio', keep_aspect_ratio)
        )
    )
    val_img_and_labels_df.rename(columns={'AbsPath': 'NpImage', 'Final Label Int': 'LabelTensor'}, inplace=True)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (list(val_img_and_labels_df['NpImage']), list(val_img_and_labels_df['LabelTensor']))
    )
    del val_ds_df
    del val_img_and_labels_df
    test_img_and_labels_df = test_ds_df[['AbsPath', 'Final Label Int']].apply(
        load_and_preprocess_image, axis=1, raw=True, result_type='reduce', args=(
            ('color_mode', color_mode), ('target_size', target_size), ('interpolation', interpolation),
            ('keep_aspect_ratio', keep_aspect_ratio)
        )
    )
    test_img_and_labels_df.rename(columns={'AbsPath': 'NpImage', 'Final Label Int': 'LabelTensor'}, inplace=True)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (list(test_img_and_labels_df['NpImage']), list(test_img_and_labels_df['LabelTensor']))
    )
    del test_ds_df
    del test_img_and_labels_df
    '''
    Batch the datasets:
    '''
    # Note: Consider enabling operation determinism if you need to debug something difficult, but this will slow
    # everything down drastically.
    train_ds = train_ds.batch(
        batch_size=batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    val_ds = val_ds.batch(
        batch_size=batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    test_ds = test_ds.batch(
        batch_size=batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    # Note: Change num_partitions to 1 to load in only Train_0, change to 2 to load in Train_0 and Train_1, etc.
    train_ds, val_ds, test_ds = load_datasets(
        color_mode='rgb', target_size=(64, 64), interpolation='nearest', keep_aspect_ratio=False, num_partitions=6,
        batch_size=32, num_images=None, train_set_size=0.6, val_set_size=0.2, test_set_size=0.2, seed=42
    )

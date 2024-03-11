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


def preprocess_image_at_path(*args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Helper method intended to be mapped over an existing :class:`tf.data.Dataset` object which contains only image file
    paths.

    .. todo:: Add preprocessing logic here. Convert to grayscale, downsample, etc.

    Args:
        *args: Variable length argument list. The inputs depend on the :class:`pd.DataFrame` object (or
          :class:`pd.Series`) that the :meth:`pd.DataFrame.apply` method is called on.
        **kwargs: Arbitrary keyword arguments. The inputs depend on the :class:`pd.DataFrame` object (or
          :class:`pd.Series`) that the :meth:`pd.DataFrame.apply` method is called on.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the image and its corresponding label.

    See Also:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply

    """
    image_abs_path = args[0][0]
    image_int_label = args[0][1]
    image = tf.io.read_file(image_abs_path, name=image_int_label)
    label = tf.convert_to_tensor(image_int_label, dtype=tf.int32)
    # .. todo:: Preprocessing logic goes here. Convert to grayscale, downsample, etc.
    return image, label


def load_and_preprocess_image(*args, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Load an image into a :class:`np.ndarray` preprocess it, return the label as an int.
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


def load_datasets(train_set_size: Optional[float] = 0.6, val_set_size: Optional[float] = 0.2, test_set_size: Optional[float] = 0.2, seed: Optional[int] = 42) -> [Dataset, Dataset, Dataset]:
    """
    Constructs TensorFlow train, val, test :class:`tensorflow.data.Dataset` s from the provided high resolution image
    training set.

    Args:
        train_set_size (Optional[float]): The percentage of the data to use as the training dataset. This value defualts
          to ``60%``.
        val_set_size (Optional[float]): The percentage of the data to use as the validation dataset. This value defaults
          to ``20%``.
        test_set_size (Optional[float]): The percentage of the data to use as the testing dataset. This value defaults
          to ``20%``.
        seed (Optional[int]): The random seed used to shuffle the data and split the dataset into training, validation,
          and testing sets. This value defaults to ``42``.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and testing datasets.

    """
    high_res_train_data_path = os.path.abspath('/usr/local/data/JustRAIGS/raw/train')
    train_data_labels_csv_path = os.path.abspath(os.path.join(high_res_train_data_path, 'JustRAIGS_Train_labels.csv'))
    assert os.path.isfile(train_data_labels_csv_path), f"Failed to find {train_data_labels_csv_path}"
    train_data_labels_df = pd.read_csv(filepath_or_buffer=train_data_labels_csv_path, delimiter=';')
    train_image_abs_file_paths = []
    image_paths_df = pd.DataFrame()
    # .. todo:: Change this back to 6 when done debugging data loading pipelines:
    num_partitions = 1
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
                # .. todo:: Remove after debugging
                if file_id == 10:
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
    val_ds_df, test_ds_df = train_test_split(
        val_ds_df, train_size=val_set_size, test_size=test_set_size, shuffle=False
    )
    '''
    At this point we have the dataframes partitioned into train, validation, and testing sets.
    Now we need to convert them to TensorFlow Datasets:
    '''
    # .. todo:: Bubble up tf.keras.utils.load_img params to method signature
    train_img_and_labels_df = train_ds_df[['AbsPath', 'Final Label Int']].apply(
        load_and_preprocess_image, axis=1, raw=True, result_type='reduce', args=(
            ('color_mode', 'rgb'), ('target_size', (64, 64)), ('interpolation', 'nearest'), ('keep_aspect_ratio', False)
        )
    )
    train_img_and_labels_df.rename(columns={'AbsPath': 'NpImage', 'Final Label Int': 'LabelTensor'}, inplace=True)
    train_ds = tf.data.Dataset.from_tensor_slices((train_img_and_labels_df['NpImage'], train_img_and_labels_df['LabelTensor']))
    logger.debug(f"ele: {list(train_ds.take(1).as_numpy_iterator())}")



    # train_ds = tf.data.Dataset.from_tensor_slices((train_ds_df['AbsPath'].to_numpy(), train_ds_df['Final Label Int'].to_numpy()))
    # train_img_and_labels_df = train_ds_df[['AbsPath', 'Final Label Int']].apply(
    #     preprocess_image_at_path, axis=1, raw=True, result_type='reduce'
    # )
    # train_img_and_labels_df.rename(columns={'AbsPath': 'ImageTensor', 'Final Label Int': 'LabelTensor'}, inplace=True)
    # train_ds = tf.data.Dataset.from_tensors((train_img_and_labels_df['ImageTensor'].to_numpy(), train_img_and_labels_df['LabelTensor'].to_numpy()))

if __name__ == '__main__':
    train_ds, val_ds, test_ds = load_datasets()

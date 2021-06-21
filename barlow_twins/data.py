from functools import partial
from pathlib import Path
from typing import Union, Tuple, List

import tensorflow as tf

import barlow_twins


def _get_image_paths(folder: Union[Path, str], image_extensions: Tuple[str] = None) -> List[Path]:
    """
    Given a base folder collects all the images (recoursively) and returns with a list of individual image paths
    """

    if image_extensions is None:
        image_extensions = ("jpg", "jpeg", "png")

    folder_path = Path(folder)

    image_list = []
    for e in image_extensions:
        image_paths = folder_path.glob(f"**/*.{e}")
        image_list.extend(list(image_paths))

    return image_list


@tf.function
def _read_image_from_path(image_path) -> tf.Tensor:
    """
    Read the image to a Tensor
    """

    image = tf.io.read_file(image_path)
    # "decode_image" is used instead of "decode_jpeg" as we support multiple image formats
    image = tf.image.decode_image(image, channels=3, dtype=tf.uint8, expand_animations=False)
    return image


@tf.function
def _make_image_pair_and_augment(image: tf.Tensor, height: int, width: int, min_crop_ratio: float,
                                 max_crop_ratio: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates the image pair (same image) and (random) augments them separately
    """

    fn = partial(barlow_twins.random_augment,
                 target_height=height,
                 target_width=width,
                 min_crop_ratio=min_crop_ratio,
                 max_crop_ratio=max_crop_ratio)
    return fn(image), fn(image)


def create_dataset(folder: Union[Path, str],
                   height: int,
                   width: int,
                   batch_size: int,
                   min_crop_ratio: float = 0.3,
                   max_crop_ratio: float = 1.0,
                   shuffle_buffer_size: int = 1000) -> Tuple[tf.data.Dataset, int]:
    """
    Creation of the tf.data.Dataset object for training purposes
    Handles the following:
        - Read the image
        - Prepare pairs of images
        - Augment image pair separately
    Output image tensor has it's values in the range of [0, 255]

    (Normalization and other "low-level" preprocessings are handled in the model itself)
    """

    image_paths = _get_image_paths(folder)
    image_paths = list(map(str, image_paths))

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(_read_image_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(partial(_make_image_pair_and_augment,
                                  height=height,
                                  width=width,
                                  min_crop_ratio=min_crop_ratio,
                                  max_crop_ratio=max_crop_ratio),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, len(image_paths)

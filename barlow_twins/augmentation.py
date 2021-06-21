import tensorflow as tf


@tf.function
def _random_prob():
    return tf.random.uniform([1], minval=0.0, maxval=1.0)[0]


@tf.function
def _random_crop(image, min_crop_ratio: float, max_crop_ratio: float):
    """
    Randomly crops an image while keeping it's original aspect ratio
    Crop size is determined by the min, max parameters and the image original size
    (e.g. new_height = original_height * random(min, max)
    """

    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], dtype=tf.float32)
    width = tf.cast(image_shape[1], dtype=tf.float32)

    # With a single crop ratio, the original aspect ratio is preserved
    crop_ratio = tf.random.uniform([1], minval=min_crop_ratio, maxval=max_crop_ratio)[0]

    new_height = tf.cast(height * crop_ratio, dtype=tf.int32)
    new_width = tf.cast(width * crop_ratio, dtype=tf.int32)

    cropped_image = tf.image.random_crop(image, (new_height, new_width, 3))
    return cropped_image


def random_augment(image, target_height: int, target_width: int, min_crop_ratio: float, max_crop_ratio: float):
    """
    Random augmentation for a single image
    Output image values are still in the range of [0, 255]
    """

    # Random cropping and resizing always applied
    image = _random_crop(image, min_crop_ratio, max_crop_ratio)
    image = tf.image.resize(image, (target_height, target_width))

    if _random_prob() > 0.5:
        image = tf.image.random_flip_left_right(image)
    if _random_prob() > 0.2:
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_saturation(image, 0.2, 1.8)
        image = tf.image.random_hue(image, 0.05)
    if _random_prob() > 0.8:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.repeat(image, repeats=3, axis=-1)
    # if _random_prob() > 0.1:
    #     # sigma is in range [0.1, 2]
    #     sigma = _random_prob() * 1.9 + 0.1
    #     image = tfa.image.gaussian_filter2d(image,filter_shape=(11, 11), sigma=sigma)

    return image

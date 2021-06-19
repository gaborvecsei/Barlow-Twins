import tensorflow as tf


def _random_prob():
    return tf.random.uniform([1], minval=0.0, maxval=1.0)[0]


def _random_crop(image, min_crop_ratio: float, max_crop_ratio: float):
    # TODO: We should random crop in a way that the crop has random size but it is a square image
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], dtype=tf.float32)
    width = tf.cast(image_shape[1], dtype=tf.float32)

    crop_height_ratio = tf.random.uniform([1], minval=min_crop_ratio, maxval=max_crop_ratio)[0]
    new_height = tf.cast(height * crop_height_ratio, dtype=tf.int32)

    crop_width_ratio = tf.random.uniform([1], minval=min_crop_ratio, maxval=max_crop_ratio)[0]
    new_width = tf.cast(width * crop_width_ratio, dtype=tf.int32)

    cropped_image = tf.image.random_crop(image, (new_height, new_width, 3))
    return cropped_image


def random_augment(image, target_height: int, target_width: int, min_crop_ratio: float, max_crop_ratio: float):
    # TODO: parameters were selected randomly and also probabilities - we should tune it
    # TODO: blurring and solarization is needed

    # TODO randomize the crop size and there should be a ratio of how much we want to crop
    # Random cropping and resizing always applied
    image = _random_crop(image, min_crop_ratio, max_crop_ratio)
    image = tf.image.resize(image, (target_height, target_width))

    if _random_prob() > 0.5:
        image = tf.image.random_flip_left_right(image)
    if _random_prob() > 0.5:
        image = tf.image.random_saturation(image, 5, 10)
    if _random_prob() > 0.5:
        image = tf.image.random_brightness(image, 0.5)
    if _random_prob() > 0.5:
        image = tf.image.random_contrast(image, 0.2, 0.5)
    if _random_prob() > 0.5:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.repeat(image, repeats=3, axis=-1)

    return image

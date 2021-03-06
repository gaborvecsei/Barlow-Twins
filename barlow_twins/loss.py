from typing import Tuple

import tensorflow as tf


def _get_off_diagonal_elements(x):
    """
    Returns non diagonal elements from an NxN matrix
    (source: https://github.com/facebookresearch/barlowtwins/blob/e6f34a01c0cde6f05da6f431ef8a577b42e94e71/main.py#L180)
    """

    n = tf.shape(x)[0]
    x = tf.reshape(x, [-1])
    x = x[:-1]
    x = tf.reshape(x, [n - 1, n + 1])
    x = x[:, 1:]
    x = tf.reshape(x, [-1])
    return x


def normalize(x, eps=1e-8):
    """
    Normalization along the batch dimension
    (eps is needed to handle when std is 0, otherwise the loss would be NaN)
    """

    m = tf.reduce_mean(x, axis=0)
    s = tf.math.reduce_std(x, axis=0)
    res = (x - m) / (s + eps)
    return res


def loss(z1, z2, _lambda: float, global_batch_size: int) -> tf.Tensor:
    """
    Loss calculation with 2 terms: Invariance term and Redundancy reduction term.
    The lambda scalar weights the lattern one
    """

    z1 = tf.cast(z1, dtype=tf.float32)
    z2 = tf.cast(z2, dtype=tf.float32)

    # Local to a single GPU
    local_batch_size = tf.cast(tf.shape(z1)[0], dtype=tf.float32)
    # Local batch size * number of GPUs
    global_batch_size = tf.cast(global_batch_size, dtype=tf.float32)

    # Normalization of the embeddings along the batch dimension, shape: (N, D)
    z1 = normalize(z1)
    z2 = normalize(z2)

    c = tf.transpose(z1) @ z2
    c = c / local_batch_size

    # This is the invariance term
    on_diag = tf.pow(tf.linalg.diag_part(c) - 1, 2)
    on_diag = tf.reduce_sum(on_diag)

    # This is the redundancy reduction term
    off_diag = tf.pow(_get_off_diagonal_elements(c), 2)
    off_diag = tf.reduce_sum(off_diag)
    off_diag = _lambda * off_diag

    loss = on_diag + off_diag

    # This is needed as we distributed the training across multiple GPUs but we already
    # averaged the correlation matrix with the "local" batch size
    loss = loss / (global_batch_size / local_batch_size)

    return loss

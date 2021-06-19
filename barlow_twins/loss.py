import tensorflow as tf


def _get_off_diagonal_elements(x):
    # source: https://github.com/facebookresearch/barlowtwins/blob/e6f34a01c0cde6f05da6f431ef8a577b42e94e71/main.py#L180
    n = tf.shape(x)[0]
    x = tf.reshape(x, [-1])
    x = x[:-1]
    x = tf.reshape(x, [n - 1, n + 1])
    x = x[:, 1:]
    x = tf.reshape(x, [-1])
    return x


def loss(z1, z2, _lambda: float):
    batch_size = tf.shape(z1)[0]
    embedding_dim = tf.shape(z1)[1]

    # Normalization of the embeddings along the batch dimension, shape: (N, D)
    z1_norm = tf.linalg.normalize(z1)[0]
    z2_norm = tf.linalg.normalize(z2)[0]

    # Compute cross correlation matrix, shape: (D, D)
    corr = tf.linalg.matmul(tf.transpose(z1_norm), z2_norm) / tf.cast(batch_size, dtype=tf.float32)

    # Loss
    corr_diff = tf.pow((corr - tf.eye(embedding_dim)), 2)
    off_diagonal = _get_off_diagonal_elements(corr_diff)
    loss = tf.reduce_sum(corr_diff) + _lambda * tf.reduce_sum(off_diagonal)

    return loss

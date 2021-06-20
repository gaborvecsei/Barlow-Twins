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
    z1 = tf.linalg.normalize(z1)[0]
    z2 = tf.linalg.normalize(z2)[0]

    # Compute cross correlation matrix, shape: (D, D)
    corr = tf.linalg.matmul(tf.transpose(z1), z2) / tf.cast(batch_size, dtype=tf.float32)

    # Loss
    corr_diff = tf.pow((corr - tf.eye(embedding_dim)), 2)
    off_diagonal = _get_off_diagonal_elements(corr_diff)
    loss = tf.reduce_sum(corr_diff) + _lambda * tf.reduce_sum(off_diagonal)

    return loss


def normalize(x, eps=1e-8):
    m = tf.reduce_mean(x, axis=0)
    s = tf.math.reduce_std(x, axis=0)
    res = (x - m) / (s + eps)
    return res


def loss2(z1, z2, _lambda: float):
    batch_size = tf.shape(z1)[0]
    embedding_dim = tf.shape(z1)[1]

    # Normalization of the embeddings along the batch dimension, shape: (N, D)
    z1 = normalize(z1) 
    z2 = normalize(z2) 

    c = tf.transpose(z1) @ z2
    c = c / tf.cast(batch_size, dtype=tf.float32)

    on_diag = tf.pow(tf.linalg.diag_part(c) - 1, 2)
    on_diag = tf.reduce_sum(on_diag)

    off_diag = tf.pow(_get_off_diagonal_elements(c), 2)
    off_diag = tf.reduce_sum(off_diag)

    loss = on_diag + _lambda * off_diag

    return loss
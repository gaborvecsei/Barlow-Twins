import tensorflow as tf

import barlow_twins


@tf.function
def train_step(model: tf.keras.models.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               image_pairs: tf.Tensor,
               _lambda: float) -> float:
    images_1 = image_pairs[0]
    images_2 = image_pairs[1]

    with tf.GradientTape() as tape:
        z1 = model(images_1, training=True)
        z2 = model(images_2, training=True)
        loss = barlow_twins.loss(z1, z2, _lambda)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss

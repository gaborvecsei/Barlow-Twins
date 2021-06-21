import tensorflow as tf

import barlow_twins


@tf.function
def train_step(model: tf.keras.models.Model, optimizer: tf.keras.optimizers.Optimizer, image_pairs: tf.Tensor,
               _lambda: float, mixed_precision: bool) -> float:
    images_1 = image_pairs[0]
    images_2 = image_pairs[1]

    with tf.GradientTape() as tape:
        z1 = model(images_1, training=True)
        z2 = model(images_2, training=True)
        loss = barlow_twins.loss(z1, z2, _lambda)
        if mixed_precision:
            scaled_loss = optimizer.get_scaled_loss(loss)

    if mixed_precision:
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    else:
        gradients = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss

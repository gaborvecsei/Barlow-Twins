from typing import Tuple

import tensorflow as tf

import barlow_twins


@tf.function
def train_step(model: tf.keras.models.Model, optimizer: tf.keras.optimizers.Optimizer, image_pairs: tf.Tensor,
               _lambda: float, mixed_precision: bool, global_batch_size: int) -> Tuple[float, float, float]:
    images_1 = image_pairs[0]
    images_2 = image_pairs[1]

    with tf.GradientTape() as tape:
        z1 = model(images_1, training=True)
        z2 = model(images_2, training=True)
        loss = barlow_twins.loss(z1, z2, _lambda, global_batch_size)
        if mixed_precision:
            scaled_loss = optimizer.get_scaled_loss(loss)

    if mixed_precision:
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    else:
        gradients = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss


@tf.function
def distributed_train_step(mirrored_strategy, model, optimizer, dist_inputs, _lambda, mixed_precision,
                           global_batch_size):
    per_replica_losses = mirrored_strategy.run(train_step,
                                               args=(model, optimizer, dist_inputs, _lambda, mixed_precision,
                                                     global_batch_size))
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return loss

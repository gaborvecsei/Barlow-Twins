import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class WarmUpCosineDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    https://arxiv.org/abs/1608.03983
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps

    def __call__(self, *args, **kwargs):
        learning_rate_base = ops.convert_to_tensor_v2(self.learning_rate_base, name="learning_rate_base")
        dtype = learning_rate_base.dtype
        total_steps = math_ops.cast(self.total_steps, dtype)
        global_step = math_ops.cast(self.global_step, dtype)
        warmup_learning_rate = math_ops.cast(self.warmup_learning_rate, dtype)
        warmup_steps = math_ops.cast(self.warmup_steps, dtype)
        hold_base_rate_steps = math_ops.cast(self.hold_base_rate_steps, dtype)

        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
            3.141592 *
            (global_step - warmup_steps - hold_base_rate_steps
             ) / tf.cast(total_steps - warmup_steps - hold_base_rate_steps, dtype=dtype)))

        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)

        self.global_step += 1

        return tf.where(global_step > total_steps, 0.0, learning_rate)

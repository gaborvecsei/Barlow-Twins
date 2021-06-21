import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import barlow_twins
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

args = barlow_twins.get_train_args()

if args.mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

experiment_folder = Path(args.base_log_folder) / args.name
if experiment_folder.is_dir():
    if args.overwrite:
        shutil.rmtree(experiment_folder)
    else:
        raise RuntimeError(f"Experiment folder already exists: {experiment_folder}")
experiment_folder.mkdir(parents=True)

(experiment_folder / "args.json").write_text(json.dumps(args.__dict__))

dataset, nb_images = barlow_twins.create_dataset(args.data,
                                                 height=args.height,
                                                 width=args.width,
                                                 batch_size=args.batch_size,
                                                 min_crop_ratio=0.8,
                                                 max_crop_ratio=1.0,
                                                 shuffle_buffer_size=1000)

model = barlow_twins.BarlowTwinsModel(input_height=args.height,
                                      input_width=args.width,
                                      projection_units=args.projector_units,
                                      load_imagenet=False)
dummy_input = np.zeros((args.batch_size, args.height, args.width, 3), dtype=np.float32)
_ = model(dummy_input)

steps_per_epoch = nb_images // args.batch_size
total_steps = args.epochs * steps_per_epoch
warmup_steps = args.lr_warmup_epochs * steps_per_epoch
print(f"Steps per epoch {steps_per_epoch}, total steps {total_steps}, warmup steps {warmup_steps}")
lr_scheduler = barlow_twins.WarmUpCosineDecayScheduler(learning_rate_base=args.learning_rate,
                                                       total_steps=total_steps,
                                                       global_step_init=0,
                                                       warmup_learning_rate=0,
                                                       warmup_steps=warmup_steps,
                                                       hold_base_rate_steps=0)
if args.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, decay=1.5e-6)
elif args.optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)
else:
    raise RuntimeError(f"Optimizer {args.optimizer} is not a valid option")

if args.mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

loss_metric = tf.keras.metrics.Mean(name="mean_loss")
tb_file_writer = tf.summary.create_file_writer(str(experiment_folder))
tb_file_writer.set_as_default()

global_step = 0

for epoch in range(args.epochs):
    print(f"Epoch {epoch} -------------")
    for step, image_pairs in enumerate(dataset):
        loss = barlow_twins.train_step(model, optimizer, image_pairs, args.lmbda, mixed_precision=args.mixed_precision)

        loss_metric(loss)
        tf.summary.scalar("loss", loss, global_step)
        tf.summary.scalar("lr", optimizer.learning_rate(global_step), global_step)

        if step % args.print_freq == 0 and step != 0:
            print(f"\tStep {step} (global step {global_step}: loss {loss_metric.result():.4f}")

        global_step += 1

    print(f"Epoch {epoch}: Loss {loss_metric.result():.4f}")

    loss_metric.reset_states()

    model.save_weights(str(experiment_folder / f"checkpoint.h5"))

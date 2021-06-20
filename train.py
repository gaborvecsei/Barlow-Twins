from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf

import barlow_twins

tf.config.run_functions_eagerly(True)

physical_devices = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PROJECTOR_DIMS = 8192
BATCH_SIZE = 64
IMAGES_FOLDER = "/data"
EPOCHS = 200
_LAMBDA = 5e-3
EXPERIMENT_NAME = "test_1"
LOG_FOLDER = "logs"

experiment_folder = Path(LOG_FOLDER) / EXPERIMENT_NAME
if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True)
else:
    # raise RuntimeError(f"Experiment folder already exists: {experiment_folder}")
    # TODO: This is only for testing
    shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True)
    pass

dataset = barlow_twins.create_dataset(IMAGES_FOLDER, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, batch_size=BATCH_SIZE,
                                      min_crop_ratio=0.3, max_crop_ratio=1.0, shuffle_buffer_size=100)
# nb_batches = tf.data.experimental.cardinality(dataset)
# print(nb_batches)

model = barlow_twins.BarlowTwinsModel(input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH,
                                      projection_units=PROJECTOR_DIMS, load_imagenet=False)
dummy_input = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
_ = model(dummy_input)

# In the paper they used the LARS optimizer
lr_scheduler = barlow_twins.WarmUpCosineDecayScheduler(learning_rate_base=1e-3,
                                                       total_steps=10000,
                                                       global_step_init=0,
                                                       warmup_learning_rate=0,
                                                       warmup_steps=500,
                                                       hold_base_rate_steps=0)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, decay=1.5e-6)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)

loss_metric = tf.keras.metrics.Mean(name="mean_loss")
tb_file_writer = tf.summary.create_file_writer(str(experiment_folder))
tb_file_writer.set_as_default()

global_steps = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} -------------")
    for step, image_pairs in enumerate(dataset):
        loss = barlow_twins.train_step(model, optimizer, image_pairs, _LAMBDA)

        loss_metric(loss)
        tf.summary.scalar("loss", loss_metric.result(), global_steps)
        tf.summary.scalar("lr", optimizer.learning_rate(global_steps), global_steps)

        if step % 50 == 0 and step != 0:
            print(f"\tStep {step}: loss {loss_metric.result():.4f}")

        global_steps += 1

    print(f"Epoch {epoch}: Loss {loss_metric.result():.4f}")

    loss_metric.reset_states()

    model.save_weights(str(experiment_folder / f"checkpoint_epoch_{epoch}.h5"))
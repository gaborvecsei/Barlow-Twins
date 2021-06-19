import matplotlib.pyplot as plt
import numpy as np

import barlow_twins

dataset = barlow_twins.create_dataset("asd", 224, 224, 4, 0.3, 0.8, 1)

for data in dataset.take(1):
    images_1 = data[0]
    images_2 = data[1]

    for i in range(4):
        img = np.hstack((images_1[i, :, :, :].numpy(), images_2[i, :, :, :].numpy())).astype(np.uint8)
        plt.imsave(f"img{i}.jpg", img)

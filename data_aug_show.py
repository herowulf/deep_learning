from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import datasets

import func

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)




fig = plt.figure()

for i in range(5):
    ax = fig.add_subplot(5,6, 6*i + 1)
    plt.imshow(train_images[i])
    if i == 0:
        ax.title.set_text('Original Images')
    plt.axis('off')

    for j in range(5):
        ax = fig.add_subplot(5,6, 6*i + 2 + j)
        plt.imshow(datagen.random_transform(train_images[i]))
        if i == 0 and j == 2:
            ax.title.set_text('Transformed Images')
        plt.axis('off')

plt.show()

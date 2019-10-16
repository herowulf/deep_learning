# excercise 2
# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers

import func

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# datashape +(32,32,3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

'Model one hidden layer as big as you can '
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),#the hidden layer
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50,
                    validation_data=(test_images, test_labels))

func.plot_history(history, 'history_Conv32_h1_1024.png')
model.save('model_Conv32_h1_1024')

test_loss, test_acc = model.evaluate(test_images, test_labels)
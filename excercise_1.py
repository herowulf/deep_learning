# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, callbacks

import func

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# datashape +(32,32,3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



'Model no hidden layers'

'Callbacks'
callback_list = [callbacks.EarlyStopping(monitor='val_accuracy',
                                        min_delta=1e-4,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.5,
                                            patience=20,
                                            min_lr=1e-6,
                                            verbose=1)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500,
                    validation_data=(test_images, test_labels), callbacks = callback_list)

func.plot_history(history, 'history/history_h0.png')



test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

'Model one hidden layers'

'Callbacks'
callback_list = [callbacks.EarlyStopping(monitor='val_accuracy',
                                        min_delta=1e-4,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.5,
                                            patience=20,
                                            min_lr=1e-6,
                                            verbose=1)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(500, activation='relu'), # hidden layer 1
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500, batch_size=2000,
                    validation_data=(test_images, test_labels), callbacks = callback_list)

func.plot_history(history, 'history/history_h1_500.png')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)



'Model two hidden layers'

callback_list = [callbacks.EarlyStopping(monitor='val_accuracy',
                                        min_delta=1e-4,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.5,
                                            patience=20,
                                            min_lr=1e-6,
                                            verbose=1)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(500, activation='relu'), # hidden layer 1
    layers.Dense(50, activation='relu'), # hidden layer 2
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500,
                    validation_data=(test_images, test_labels), callbacks = callback_list)

func.plot_history(history, 'history/history_h1_500_h2_50.png')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

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
callback_list = [callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=1e-3,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
             callbacks.ModelCheckpoint('model_h0.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='auto',
                                       period=1,
                                       verbose=0)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500,
                    validation_data=(test_images, test_labels))

func.plot_history(history, 'history_h0.png')



test_loss, test_acc = model.evaluate(test_images, test_labels)

'Model one hidden layers'

'Callbacks'
callback_list = [callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=1e-3,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
             callbacks.ModelCheckpoint('model_h1_500.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='auto',
                                       period=1,
                                       verbose=0)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(500, activation='relu'), # hidden layer 1
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500,
                    validation_data=(test_images, test_labels))

func.plot_history(history, 'history_h1_500.png')

test_loss, test_acc = model.evaluate(test_images, test_labels)



'Model two hidden layers'

callback_list = [callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=1e-3,
                                        patience=30,
                                        verbose=0,
                                        mode='auto'),
             callbacks.ModelCheckpoint('model_h1_500_h2_50.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='auto',
                                       period=1,
                                       verbose=0)]

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(500, activation='relu'), # hidden layer 1
    layers.Dense(50, activation='relu'), # hidden layer 2
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500,
                    validation_data=(test_images, test_labels))

func.plot_history(history, 'history_h1_500_h2_50.png')

test_loss, test_acc = model.evaluate(test_images, test_labels)

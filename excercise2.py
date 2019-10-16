# excercise 2
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


'Model one hidden layer as big as you can '
steps = np.linspace(100, 10000, num=20)
steps.astype(int)

for i in range(len(steps)):
    model = models.Sequential([
        layers.Flatten(),
        layers.Dense(steps[i], activation='relu'),#the hidden layer
        layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=2,
                        validation_data=(test_images, test_labels),
                        callbacks=callback_list)
    plotname = 'history_h1_big_' + str(steps[i].astype(int)) + '.png'
    func.plot_history(history, plotname)
    # model.save(plotname)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

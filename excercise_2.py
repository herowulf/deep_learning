# excercise 2
# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import func

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# datashape +(32,32,3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

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

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


'Model one hidden layer as big as you can '
steps = np.linspace(100, 1000, num=10, dtype='int32')

record = []

for step in steps:
    model = models.Sequential([
        layers.Flatten(),
        layers.Dense(step, activation='relu'),#the hidden layer
        layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('Start training with {} neurons'.format(step))
    batch_size = 1000
    history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        steps_per_epoch=len(train_images) / batch_size, epochs=500,
                        callbacks=callback_list, validation_data = (test_images, test_labels))

    # history = model.fit(train_images, train_labels,
    #                     epochs=1000,
    #                     batch_size=2000,
    #                     validation_data=[test_images, test_labels],
    #                     callbacks=callback_list)

    func.plot_history(history, 'history/history_h1_{}_big_imagegen.png'.format(step))
    # model.save(plotname)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    record.append([step, test_loss, test_acc])

    print('Finished training {}'.format(record[-1]))

print(record)

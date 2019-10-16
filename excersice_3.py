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
            callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            factor=0.5,
                                            patience=20,
                                            min_lr=1e-6,
                                            verbose=1)]


record = []

for activ in ['relu', 'tanh']:
    for opt in ['adam', 'SGD', 'RMSprop']:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=activ, input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=activ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=activ))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=activ))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=500,
                            validation_data=(test_images, test_labels),
                            callbacks=callback_list)

        func.plot_history(history, 'history_conv_{}_{}.png'.format(opt, activ))

        print('Training {} {} done!'.format(opt, activ))

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        record.append([test_loss, test_acc])

print(record)

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import func

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# datashape +(32,32,3)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


callback_list = [callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=1e-3,
                                        patience=20,
                                        verbose=0,
                                        mode='auto'),
            callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            factor=0.5,
                                            patience=10,
                                            min_lr=1e-6,
                                            verbose=1)]

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=1024),
                    steps_per_epoch=len(train_images) / 1000, epochs=500,
                    callbacks=callback_list, validation_data = (test_images, test_labels))


# history = model.fit(train_images, train_labels, epochs=500,
#                     validation_data=(test_images, test_labels),
#                     callbacks=callback_list)

func.plot_history(history, 'history/history_conv_best.png')

print('Training done!')

test_loss, test_acc = model.evaluate(test_images, test_labels)

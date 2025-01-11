import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.utils import img_to_array, load_img
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tensorflow.dtensor.python import num_clients

project_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(project_dir, "dataset", "fruits", "train")
test_path = os.path.join(project_dir, "dataset", "fruits", "test")

categories = os.listdir(train_path)
categories.sort()
num_classes = len(categories)

BatchSize = 64
inputSize = 250

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(inputSize,inputSize,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2048, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3
)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(inputSize,inputSize),
    batch_size=BatchSize,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(inputSize,inputSize),
    batch_size=BatchSize,
    color_mode="rgb",
    class_mode="categorical"
)

stepsPerEpoch = np.ceil(train_generator.samples / BatchSize)
validationSteps = np.ceil(test_generator.samples / BatchSize)

#stop_early = EarlyStopping(monitor="val_accuracy", patience=10, min_delta=0.001, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=stepsPerEpoch,
    epochs=150,
    validation_data=test_generator,
    validation_steps=validationSteps,
    #callbacks=[stop_early]
)

# Zapisanie modelu
model.save(project_dir + "/fruits_classifier_with_conv_layers.h5")

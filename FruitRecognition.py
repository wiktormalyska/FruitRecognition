import tensorflow as tf
from tensorflow.python import keras
from keras import layers, models, applications
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

dataset_path_training = "dataset/fruits-360/Training"
dataset_path_testing = "dataset/fruits-360/Test"

data_gen = ImageDataGenerator(
    rescale=1.0 /255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)

train_data = data_gen.flow_from_directory(
    dataset_path_training,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_data = data_gen.flow_from_directory(
    dataset_path_testing,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

base_model = applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")

model.save("fruits_classifier.h5")
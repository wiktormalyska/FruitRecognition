import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from config import K_FOLDS, IMAGE_SIZE, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, LEARNING_RATE

#
# Definicja Danych
#

project_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(project_dir, "dataset", "fruits-360", "train")
test_path = os.path.join(project_dir, "dataset", "fruits-360", "test")

categories = os.listdir(train_path)
num_classes = len(categories)

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(train_path, category)
    for img_file in os.listdir(category_path):
        image_paths.append(os.path.join(category_path, img_file))
        labels.append(category)

#
# Definicja modelu
#

def build_model(num_classes_prop):
    model_build = Sequential()
    model_build.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model_build.add(BatchNormalization())
    model_build.add(MaxPooling2D())
    model_build.add(Dropout(0.3))

    model_build.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model_build.add(BatchNormalization())
    model_build.add(MaxPooling2D())
    model_build.add(Dropout(0.3))

    model_build.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
    model_build.add(BatchNormalization())
    model_build.add(MaxPooling2D())
    model_build.add(Dropout(0.4))

    model_build.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
    model_build.add(BatchNormalization())
    model_build.add(MaxPooling2D())
    model_build.add(Dropout(0.4))

    model_build.add(Flatten())
    model_build.add(Dense(1024, activation="relu"))
    model_build.add(Dropout(0.5))
    model_build.add(Dense(512, activation="relu"))
    model_build.add(Dense(num_classes_prop, activation="softmax"))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model_build.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model_build

#
# Przetwarzanie obrazów
#

BatchSize = BATCH_SIZE
inputSize = IMAGE_SIZE

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.4,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale= 1./255)

#
# Definicja kroswalidacji
#
K = K_FOLDS

kf = KFold(n_splits=K, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\n🔹 Trenowanie modelu dla Fold {fold+1}/{K}")

    train_images = [image_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]

    train_generator = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=IMAGE_SIZE,
        batch_size=BatchSize,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=IMAGE_SIZE,
        batch_size=BatchSize,
        class_mode="categorical"
    )

    model = build_model(num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        # callbacks=[early_stopping]
    )

    scores = model.evaluate(val_generator)
    fold_accuracies.append(scores[1])

    model.save(f"{project_dir}/fruits_classifier_fold{fold+1}.h5")

print(f"\n✅ Średnia dokładność K-Fold: {np.mean(fold_accuracies):.4f}")

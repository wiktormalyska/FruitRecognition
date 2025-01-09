from sys import prefix

from tensorflow.python import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
model = load_model("fruits_classifier.h5")

test_img_path = "img/apple.jpg"

img = image.load_img(test_img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions, axis=1)

class_names = {v: k for k, v in model.class_indices.itemd()}
predicted_label = class_names[predicted_class[0]]

print(f"Przewidywanie: {predicted_label}")
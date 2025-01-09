from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

model = load_model("fruits_classifier.h5")

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

img_path = 'img/apple2.jpg'
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])
predicted_class = class_labels[predicted_index]

print(f"Model przewiduje, że obraz to: {predicted_class}")

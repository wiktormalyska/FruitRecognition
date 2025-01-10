from cgitb import reset

import tensorflow as tf
import os
from keras.utils import img_to_array, load_img
import numpy as np
import cv2

model = tf.keras.models.load_model("fruits_classifier.h5")
# model = tf.keras.models.load_model("fruits_classifier_adam.h5")

project_dir = os.path.dirname(os.path.abspath(__file__))


source_folder = os.path.join(project_dir, "dataset", "fruits-360", "Test")
categories = os.listdir(source_folder)
categories.sort()

def prepare_image(path_for_image):
    img = load_img(path_for_image, target_size=(100,100))
    img_result = img_to_array(img)
    print(img_result.shape)
    img_result = np.expand_dims(img_result, axis=0)
    print(img_result.shape)
    img_result = img_result/255
    return img_result

def predict_and_display(image_path):

    image_for_model = prepare_image(image_path)
    result_array = model.predict(image_for_model, verbose=1)
    answer = np.argmax(result_array, axis=1)[0]
    text = categories[answer]

    print(f"Predicted: {text}")


    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(f"Prediction - {text}", img)

def process_images_in_folder(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            predict_and_display(image_path)

img_folder_path = os.path.join(project_dir, "img")
process_images_in_folder(img_folder_path)

cv2.waitKey(0)
cv2.destroyAllWindows()
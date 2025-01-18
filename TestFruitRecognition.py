import os
import tensorflow as tf
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import cv2
from keras.utils import img_to_array, load_img
from PIL import Image, ImageTk
from config import K_FOLDS, IMAGE_SIZE

K = K_FOLDS
project_dir = os.path.dirname(os.path.abspath(__file__))

models = []
for fold in range(K):
    model_path = f"{project_dir}/fruits_classifier_fold{fold + 1}.h5"
    model = tf.keras.models.load_model(model_path)
    models.append(model)

source_folder = os.path.join(project_dir, "dataset", "fruits-360", "train")
categories = os.listdir(source_folder)
categories.sort()


def prepare_image(path_for_image):
    img = load_img(path_for_image, target_size=IMAGE_SIZE)
    img_result = img_to_array(img)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255
    return img_result


def generate_heatmap(model, image_for_model, last_conv_layer_name="conv2d"):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_for_model)
        loss = predictions[:, np.argmax(predictions)]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


def predict_and_display(image_path):
    image_for_model = prepare_image(image_path)
    results = []

    for predict_model in models:
        result_array = predict_model.predict(image_for_model, verbose=0)
        results.append(result_array)

    avg_result = np.mean(results, axis=0)
    answer = np.argmax(avg_result, axis=1)[0]
    text = categories[answer]

    heatmap = generate_heatmap(models[0], image_for_model)
    overlay = overlay_heatmap(image_path, heatmap)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay[:, :, ::-1])
    plt.axis("off")
    plt.title(f"Predykcja: {text}")
    plt.show()

    return text, image_path


def update_image_and_text():
    global current_index, image_files, correct_predictions
    if current_index >= len(image_files):
        accuracy = (correct_predictions / len(image_files)) * 100
        text_label.config(text=f"Wynik: {accuracy:.2f}% poprawnych predykcji")
        image_label.config(image='')
        return

    image_path = os.path.join(img_folder_path, image_files[current_index])
    text, _ = predict_and_display(image_path)
    img = Image.open(image_path)
    img_resized = img.resize(IMAGE_SIZE)
    img_tk = ImageTk.PhotoImage(img_resized)

    image_label.config(image=img_tk)
    image_label.image = img_tk
    text_label.config(text=text)


def evaluate_prediction(correct):
    global current_index, correct_predictions
    if correct:
        correct_predictions += 1
    current_index += 1
    update_image_and_text()


project_dir = os.path.dirname(os.path.abspath(__file__))
img_folder_path = os.path.join(project_dir, "dataset", "fruits", "predict")
image_files = [f for f in os.listdir(img_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]
image_files.sort()

current_index = 0
correct_predictions = 0

root = tk.Tk()
root.title("Galeria Zdjęć")
root.geometry("800x600")

frame = tk.Frame(root)
frame.pack(pady=50)

text_label = tk.Label(frame, text="Opis zdjęcia", font=("Arial", 16))
text_label.pack()

image_label = tk.Label(frame)
image_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(side="bottom", pady=20)

ok_button = tk.Button(button_frame, text="OK", command=lambda: evaluate_prediction(True))
ok_button.pack(side="left", padx=10)

not_ok_button = tk.Button(button_frame, text="Nie OK", command=lambda: evaluate_prediction(False))
not_ok_button.pack(side="right", padx=10)

update_image_and_text()

root.mainloop()

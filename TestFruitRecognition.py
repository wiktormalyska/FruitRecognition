import os
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import PhotoImage
from keras.utils import img_to_array, load_img
from PIL import Image, ImageTk

# Ładujemy model
model = tf.keras.models.load_model("fruits_classifier_with_conv_layers.h5")

project_dir = os.path.dirname(os.path.abspath(__file__))

source_folder = os.path.join(project_dir, "dataset", "fruits", "train")
categories = os.listdir(source_folder)
categories.sort()

def prepare_image(path_for_image):
    # Zmieniamy rozmiar obrazu do (250, 250), bo taki rozmiar oczekuje model
    img = load_img(path_for_image, target_size=(250, 250))  # Zmieniamy na 250x250
    img_result = img_to_array(img)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255  # Normalizacja
    return img_result


def predict_and_display(image_path):
    image_for_model = prepare_image(image_path)
    result_array = model.predict(image_for_model, verbose=1)
    answer = np.argmax(result_array, axis=1)[0]
    text = categories[answer]
    return text, image_path

def update_image_and_text(image_path, text):
    # Używamy Pillow do załadowania obrazu
    img = Image.open(image_path)
    img_resized = img.resize((250, 250))  # Zmieniamy rozmiar na 250x250

    # Konwertujemy obraz do formatu, który Tkinter potrafi obsłużyć
    img_tk = ImageTk.PhotoImage(img_resized)

    # Wyświetlamy obraz
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Przechowujemy referencję do obrazu, aby nie został usunięty przez garbage collector

    # Wyświetlamy tekst
    text_label.config(text=text)
# Wczytanie folderu z obrazami
img_folder_path = os.path.join(project_dir, "dataset", "fruits", "predict")
image_files = [f for f in os.listdir(img_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]
image_files.sort()

# Globalne zmienne do śledzenia bieżącego obrazu
current_index = 0

def show_next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    image_path = os.path.join(img_folder_path, image_files[current_index])
    text, _ = predict_and_display(image_path)
    update_image_and_text(image_path, text)

def show_prev_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    image_path = os.path.join(img_folder_path, image_files[current_index])
    text, _ = predict_and_display(image_path)
    update_image_and_text(image_path, text)

# Tworzymy GUI
root = tk.Tk()
root.title("Galeria Zdjęć")
root.geometry("800x600")

# Ramka na zdjęcie i tekst
frame = tk.Frame(root)
frame.pack(pady=50)

# Tekst nad zdjęciem
text_label = tk.Label(frame, text="Opis zdjęcia", font=("Arial", 16))
text_label.pack()

# Miejsce na zdjęcie
image_label = tk.Label(frame)
image_label.pack()

# Ramka na przyciski
button_frame = tk.Frame(root)
button_frame.pack(side="bottom", pady=20)

# Przyciski na dole
prev_button = tk.Button(button_frame, text="Poprzednie zdjęcie", command=show_prev_image)
prev_button.pack(side="left", padx=20)

next_button = tk.Button(button_frame, text="Następne zdjęcie", command=show_next_image)
next_button.pack(side="right", padx=20)

# Pokazanie pierwszego obrazu przy uruchomieniu
show_next_image()

# Uruchamiamy pętlę główną aplikacji
root.mainloop()

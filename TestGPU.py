import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

import os

project_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(project_dir, "dataset", "fruits", "train")
folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
print("Foldery w katalogu treningowym:", folders)

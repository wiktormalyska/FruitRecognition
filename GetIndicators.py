from tensorflow.keras.models import load_model

# Załaduj model
model = load_model("fruits_classifier.h5")

print(model.class_indices)
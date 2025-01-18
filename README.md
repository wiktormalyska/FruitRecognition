# 🍎 Fruit Classification with CNN and K-Fold Cross-Validation  

This project is a deep learning-based fruit classification system using Convolutional Neural Networks (CNNs).  
The model is trained and evaluated using K-Fold Cross-Validation to improve generalization.  
Additionally, a visualization tool is provided to interpret model predictions with heatmaps.  

## 📂 Project Structure  

```plaintext
📦 fruit-classification
├── 📂 dataset
│   ├── 📂 fruits-360
│   │   ├── 📂 train
│   │   ├── 📂 test
│   │   ├── 📂 predict
├── 📝 config.py
├── 🏋️ FruitRecognition.py
├── 🖼️ TestFruitRecognition.py
├── 📊 TestGPU.py
└── 📜 README.md
```

## ⚙️ Configuration (config.py)
```python
K_FOLDS = 10
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 0.0001
```

## 🏋️ Training (FruitRecognition.py)
The `FruitRecognition.py` script:

- Loads and augments fruit images from the Fruits-360 dataset
- Defines a CNN model for image classification
- Applies K-Fold Cross-Validation (K=10) to improve generalization
- Uses Early Stopping to prevent overfitting
- Saves trained models for each fold

### Model Architecture
```python
model.add(Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
```

### Training Process
```python
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
```
The trained models are saved as:
```plaintext
fruits_classifier_fold1.h5
fruits_classifier_fold2.h5
...
fruits_classifier_fold10.h5
```

## 🔍 Prediction & Heatmap Visualization (TestFruitRecognition.py)
The `TestFruitRecognition.py` script:

- Loads the trained K models
- Predicts fruit categories from new images
- Generates Class Activation Maps (CAMs) to visualize important features
- Uses Tkinter GUI for interactive evaluation

### Example Prediction Flow
```python
image = load_img(image_path, target_size=IMAGE_SIZE)
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

predictions = [model.predict(image) for model in models]
avg_prediction = np.mean(predictions, axis=0)
```

### Heatmap Generation
```python
def generate_heatmap(model, image):
    grad_model = tf.keras.models.Model([model.input], [model.get_layer("conv2d").output, model.output])
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(image)
        loss = prediction[:, np.argmax(prediction)]
    grads = tape.gradient(loss, conv_output)
    heatmap = np.maximum(conv_output[0] @ grads[..., np.newaxis], 0)
    heatmap /= np.max(heatmap)
    return heatmap
```

### Interactive GUI
The GUI helps in manually verifying predictions.
It allows users to accept (OK) or reject (Not OK) the predicted fruit category.
```python
root = tk.Tk()
root.title("Fruit Classification")
text_label = tk.Label(root, text="Prediction:")
text_label.pack()
image_label = tk.Label(root)
image_label.pack()
ok_button = tk.Button(root, text="OK", command=lambda: evaluate(True))
not_ok_button = tk.Button(root, text="Not OK", command=lambda: evaluate(False))
ok_button.pack()
not_ok_button.pack()
root.mainloop()
```

### Results
After evaluating predictions, the script calculates the accuracy based on user feedback.
```python
accuracy = (correct_predictions / len(image_files)) * 100
print(f"Final Accuracy: {accuracy:.2f}%")
```

## 🚀 How to Run
### Recommended Python 3.9

### 1️⃣ Install Dependencies
```sh
pip install tensorflow keras numpy matplotlib opencv-python pillow
```
### 2️⃣ Train the Model
```sh
python FruitRecognition.py
```
### 3️⃣ Run the Prediction GUI
```sh
python TestFruitRecognition.py
```
## 🏆 Acknowledgments
- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- TensorFlow & Keras Community
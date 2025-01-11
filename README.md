## Fruit Recognition

---

### Dataset: [Kaggle Fruit Classification (10 Classes)](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)

#### This dataset contains 10 classes of fruits. The dataset contains 10 folders, one for each class of fruit. The folder names are:
1. Apple
2. Avocado
3. Banana
4. Cherry
5. Kiwi
6. Mango
7. Orange
8. Pineapple
9. Strawberries
10. Watermelon

#### Each folder contains images of the respective fruit. The images are of various sizes and shapes. The dataset is not very large, but it is sufficient for training a simple model.

---

### Model Overview

The fruit recognition model was built using the TensorFlow framework and the Keras API. It is designed to classify images into one of 10 fruit categories based on the provided dataset.

#### Model Architecture
The model uses a Convolutional Neural Network (CNN) architecture, which is highly effective for image classification tasks. 

**Key components of the model architecture:**
1. **Input Layer:**
   - Input image size: `250x250x3` (RGB images with 250x250 resolution).

2. **Convolutional Blocks:**
   - 4 convolutional layers with filter sizes increasing from 32 to 256.
   - Each convolutional layer uses a kernel size of `3x3` and ReLU activation.
   - Batch Normalization is applied after each convolutional layer to stabilize and accelerate training.
   - MaxPooling layers are used to reduce spatial dimensions.

3. **Regularization:**
   - Dropout layers (30% to 40%) are added to reduce overfitting by randomly deactivating neurons during training.

4. **Fully Connected Layers:**
   - A Flatten layer transforms the 2D feature maps into a 1D vector.
   - Two dense layers with 4096 and 2048 neurons, respectively, followed by ReLU activation.
   - A final dense layer with 10 neurons (one for each fruit class) and softmax activation for output probabilities.

5. **Optimizer and Loss Function:**
   - Loss function: Categorical Crossentropy, suitable for multi-class classification.
   - Optimizer: SGD (Stochastic Gradient Descent) with a learning rate optimized during training.
   - Metric: Accuracy.

---

### Training Process
The model was trained on the training portion of the dataset using the following parameters:
- **Batch Size:** 64
- **Epochs:** 150
- **Data Augmentation:** 
  - Applied to the training data to increase diversity:
    - Rescaling pixel values to [0, 1].
    - Horizontal and vertical flips.
    - Random zoom and shear transformations.
- **Validation Data:** The test dataset was used for validation during training.

The training achieved an accuracy of **88% after 150 epochs**, indicating a high level of performance for this dataset.

---

### Model Evaluation
The trained model was saved in the `h5` format as `fruits_classifier_with_conv_layers.h5`. It can be used to predict the fruit type of new images by preprocessing them to match the input size (250x250) and normalizing pixel values.

---

### Potential Applications
- **Grocery Store Automation:** Automatically identify fruits at checkout counters.
- **Educational Tools:** Teach children to recognize different types of fruits.
- **Quality Control:** Use in food processing to detect and classify fruits.

This model can serve as a foundational approach for more complex image recognition tasks by further tuning and using larger datasets.

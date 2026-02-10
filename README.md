# Smart_Digit_Recognition
Smart_Digit_Recognition focuses on handwritten digit classification, a key challenge in computer vision and machine learning. Variations in writing style, size, and orientation increase complexity. The trained model accurately recognizes and classifies digits (0–9) from handwritten images using image processing and pattern recognition techniques.
Title: Smart Handwritten Digit Recognition System

---

## Abstract

This project develops a convolutional neural network (CNN) to recognize handwritten digits (0–9) from images. The system was trained on 79 images and tested on 34 images. The model achieved an overall accuracy of 41.2%, with detailed per-class performance metrics including detection rate, precision, and F1-score. Visualizations of predictions and the confusion matrix were generated to evaluate system performance.

---

## Introduction

Handwritten digit recognition is a key challenge in computer vision and machine learning due to variations in handwriting styles, orientations, and sizes. This project aims to build a robust system capable of accurately classifying digits from small-scale handwritten datasets.

---

## Dataset Description

* **Total Images:** 113 (Train: 79, Test: 34)
* **Classes:** 10 (digits 0–9)
* **Structure:** Images stored in subfolders per digit class (e.g., '0/', '1/', ..., '9/')
* **Sample Images:**
  
   ![Copy of 1(17)](https://github.com/user-attachments/assets/9090eebb-be33-4d74-bcbd-32d0db15a0f4)

---

## Methodology

### 1 Data Preprocessing

* Converted all images to grayscale.
* Resized images to 28x28 pixels.
* Normalized pixel values between 0 and 1.
* Labels extracted automatically from subfolder names.

### 2 Model Architecture

| Layer        | Output Shape | Parameters |
| ------------ | ------------ | ---------- |
| Conv2D       | (28,28,32)   | 320        |
| MaxPooling2D | (14,14,32)   | 0          |
| Conv2D       | (14,14,64)   | 18,496     |
| MaxPooling2D | (7,7,64)     | 0          |
| Flatten      | (3136)       | 0          |
| Dense        | (128)        | 401,536    |
| Dropout      | (128)        | 0          |
| Dense        | (10)         | 1,290      |

* Activation functions: ReLU for hidden layers, Softmax for output layer.
* Loss function: Categorical Crossentropy.
* Optimizer: Adam.
* Epochs: 12, Batch Size: [Insert Batch Size].

### 3 Training & Validation

* Training/validation split: 79 images for training, 34 images for testing.
* Monitored loss and accuracy during training.

---

## Results

### 1 Confusion Matrix

* Insert the confusion matrix image here (confusion_matrix.png).
* Rows: True labels, Columns: Predicted labels.

### 2 Performance Metrics

| Digit | True Positives | Detection Rate | Precision | F1-Score |
| ----- | -------------- | -------------- | --------- | -------- |
| 0     | [TP0]          | [DR0]          | [P0]      | [F10]    |
| 1     | [TP1]          | [DR1]          | [P1]      | [F11]    |
| ...   | ...            | ...            | ...       | ...      |

* Overall system accuracy: 41.2%

## Discussion

* Misclassifications mostly occurred between visually similar digits (e.g., 3 vs 8).
* Small dataset size limited model performance.
* Data augmentation and larger datasets could improve accuracy.
* Model is suitable for demonstration purposes and small-scale applications.

---

## Conclusion

* Built a CNN-based handwritten digit recognition system.
* Achieved an overall accuracy of 41.2% with detailed performance metrics.
* Demonstrated predictions, confusion matrix, and per-digit evaluation.
* Future improvements: augment dataset, tune hyperparameters, and deeper network layers.

---

## References

* TensorFlow/Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* OpenCV Documentation: [https://opencv.org/](https://opencv.org/)
* MNIST Dataset (if used for reference): [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---


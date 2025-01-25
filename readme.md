# Handwritten Digit Recognition with Neural Networks

This project demonstrates a neural network-based approach to handwritten digit recognition inspired by the research paper "Handwritten Digit Recognition with a Back-Propagation Network" by Yann LeCun et al. It uses a convolutional neural network (CNN) to classify grayscale images of handwritten digits into one of 10 classes (0-9).

---

## Features
- **Dataset**: MNIST (grayscale images of handwritten digits).
- **Preprocessing**: Images are resized to 16x16 pixels and normalized to the range [-1, 1].
- **Network Architecture**:
  - Convolutional layers for feature extraction.
  - Subsampling (pooling) layers for dimensionality reduction.
  - Fully connected layers for classification.
- **Output**: Probabilities for 10 classes (digits 0-9).

---

## Getting Started

### Prerequisites
Ensure you have Python installed on your system. Install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### File Structure
- **`main.py`**: Contains the implementation of the neural network.
- **`requirements.txt`**: Lists dependencies for the project.
- **`README.md`**: Provides an overview and instructions.

---

## Workflow

### 1. Data Loading
- Load the MNIST dataset using TensorFlow/Keras.
- Split data into training and testing sets.

### 2. Preprocessing
- Resize images to 16x16 pixels.
- Normalize pixel values to the range [-1, 1].
- One-hot encode the labels for multi-class classification.

### 3. Network Architecture
The model consists of:
- **Input layer**: Accepts 16x16 grayscale images.
- **Convolutional layers**: Detect local patterns (features).
- **Subsampling layers**: Perform local averaging to reduce dimensionality.
- **Fully connected layers**: Classify the extracted features into 10 classes.

### 4. Training
- Use categorical cross-entropy as the loss function.
- Train using the Adam optimizer.
- Monitor validation accuracy to avoid overfitting.

### 5. Evaluation
- Test the model on unseen data.
- Compute accuracy and visualize predictions.

### 6. Deployment
- Save the trained model.
- Use it to classify new handwritten digits.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script.

4. (Optional) Visualize predictions:
   Modify the script to load a test image and visualize the prediction.

---

## Example Output
- **Accuracy**: ~98% on the MNIST test set.
- **Predictions**: The model predicts digits correctly for most handwritten samples. For ambiguous cases, it uses confidence thresholds to make decisions.

---

## Improvements
- Add a rejection mechanism for low-confidence predictions.
- Train on a more diverse dataset like USPS or custom data.
- Deploy the model on edge devices using TensorFlow Lite.

---

## References
- LeCun, Yann et al., "Handwritten Digit Recognition with a Back-Propagation Network," NIPS 1989.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

Feel free to contribute to this project by submitting pull requests or reporting issues!

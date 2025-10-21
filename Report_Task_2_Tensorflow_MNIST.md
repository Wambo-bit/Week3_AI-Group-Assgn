# 🧠 MNIST CNN Classification Report

## 📘 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**.  
The script demonstrates the full deep learning workflow — from data preprocessing to model evaluation and visualization.

---

## ⚙️ Step-by-Step Workflow

### 1️⃣ Import Libraries
The script imports the following Python libraries:
- `tensorflow` for building and training the neural network
- `matplotlib` for plotting results and visualizations
- `numpy` for numerical operations

### 2️⃣ Load & Preprocess Data
- Loads the **MNIST dataset** using `tf.keras.datasets.mnist.load_data()`  
- Normalizes pixel values to the range `[0, 1]`  
- Reshapes data into `(28, 28, 1)` to include the grayscale channel  

```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
```

---

## 🧩 Model Architecture (CNN)

| Layer | Type | Output Shape | Parameters | Description |
|--------|------|--------------|-------------|--------------|
| 1 | Conv2D (32, 3x3) | (None, 26, 26, 32) | 320 | Extracts low-level features |
| 2 | MaxPooling2D (2x2) | (None, 13, 13, 32) | 0 | Reduces spatial dimensions |
| 3 | Conv2D (64, 3x3) | (None, 11, 11, 64) | 18,496 | Captures deeper features |
| 4 | MaxPooling2D (2x2) | (None, 5, 5, 64) | 0 | Further reduces dimensions |
| 5 | Flatten | (None, 1600) | 0 | Converts feature maps into a vector |
| 6 | Dense (128, relu) | (None, 128) | 204,928 | Fully connected layer |
| 7 | Dense (10, softmax) | (None, 10) | 1,290 | Output probabilities |

**Total Parameters:** 225,034  
**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Metric:** Accuracy  

---

## 📊 Model Training
The model is trained for **5 epochs** using a batch size of **128** with a **10% validation split**.

```python
history = model.fit(
    x_train, y_train, 
    epochs=5, batch_size=128, 
    validation_split=0.1
)
```

Expected results:
- **Training Accuracy:** ~99%  
- **Validation Accuracy:** ~98%  
- **Test Accuracy:** ~98–99%

---

## 🧪 Evaluation
After training, the model is evaluated on the MNIST test set (10,000 images).

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
```

✅ **Expected Test Accuracy:** Around **0.985–0.990**  

---

## 🖼️ Visualizations

### Sample Predictions
Displays 5 test images with their **predicted** and **true** labels:

![Sample Predictions Placeholder](https://via.placeholder.com/700x250?text=Sample+Prediction+Visualization)

---

### (Optional) Accuracy and Loss Graphs *(for future use)*
You can later include plots of training vs validation accuracy/loss here.

| Metric | Graph Placeholder |
|---------|--------------------|
| **Accuracy** | ![Accuracy Graph Placeholder](https://via.placeholder.com/400x200?text=Accuracy+Graph) |
| **Loss** | ![Loss Graph Placeholder](https://via.placeholder.com/400x200?text=Loss+Graph) |

---

## 💪 Strengths

✅ Clear and well-documented structure  
✅ Proper normalization and model setup  
✅ Visualization of predictions included  
✅ Uses modern TensorFlow API  
✅ Easy to extend or save model (`model.save()`)  

---

## ⚠️ Improvement Suggestions

| Area | Suggestion | Benefit |
|------|-------------|----------|
| Epochs | Increase from 5 → 10–15 | Better convergence |
| Callbacks | Use `EarlyStopping`, `ModelCheckpoint` | Prevent overfitting |
| Metrics | Add confusion matrix & classification report | Better insight per class |
| Data Augmentation | Add rotation/shift/zoom | Improves robustness |
| Modularity | Split into functions | Easier to maintain |

---

## 🧾 Summary

**Dataset:** MNIST (70,000 handwritten digit images)  
**Framework:** TensorFlow/Keras  
**Model Type:** CNN  
**Training Time:** ~2–3 minutes on CPU  
**Accuracy:** ~98–99%  
**File:** `task_2_tensorflow_mnist.py`

---

**Author:** *Emmanuel Kichinda*  
**Date:** *21st.October, 2025*  
**Purpose:** PLP Deep Learning Task 2 – TensorFlow MNIST Classification  

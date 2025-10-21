# File: task_2_tensorflow_mnist.py

# 1. Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# 2. Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include the channel dimension (1 for grayscale)
# The expected shape is (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print("-" * 30)

# 3. Build the CNN model
model = Sequential([
    # First convolution layer: 32 filters, 3x3 kernel size, relu activation
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer to downsample
    MaxPooling2D(pool_size=(2, 2)),
    # Second convolution layer
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    # Second max pooling layer
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten the 2D feature maps into a 1D vector
    Flatten(),
    # Dense layer for classification
    Dense(128, activation='relu'),
    # Output layer with 10 units (for 10 digits) and softmax activation
    Dense(10, activation='softmax')
])

# 4. Compile the model
# Using Adam optimizer, sparse categorical crossentropy for integer labels, and tracking accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("-" * 30)

# 5. Train the model
print("Training the model...")
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
print("Model training complete.")
print("-" * 30)

# 6. Evaluate the model
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
print("-" * 30)

# 7. Visualize predictions on 5 sample images
print("Visualizing predictions...")
predictions = model.predict(x_test)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# To save the model for the bonus task
# model.save('mnist_cnn_model.h5')
# print("Model saved as mnist_cnn_model.h5")
# File: app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

# Function to load and cache the model
@st.cache_resource
def load_model():
    """Loads the pre-trained MNIST model."""
    try:
        # Try to load the trained model from file
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the uploaded image
def preprocess_image(image):
    """Converts uploaded image to the format expected by the model."""
    # Convert image to grayscale
    img = image.convert('L')
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    # Convert image to numpy array
    img_array = np.array(img)
    # Invert colors (model was trained on white digits on black background)
    img_array = 255 - img_array
    # Normalize the image
    img_array = img_array / 255.0
    # Reshape to (1, 28, 28, 1) to match model's input shape
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# --- Main App ---
model = load_model()

# Title and description
st.title("🧠 MNIST Handwritten Digit Recognizer")
st.markdown("Upload an image of a handwritten digit (0-9), and the AI will predict what it is.")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Upload your image here...", type=["png", "jpg", "jpeg"])

if model and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Uploaded Image")
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner('Analyzing the image...'):
            # Preprocess the image and make a prediction
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.success(f"**Predicted Digit: {predicted_digit}**")
            st.info(f"**Confidence: {confidence:.2%}**")

            # Display prediction probabilities
            st.subheader("Prediction Probabilities")
            st.bar_chart(prediction.flatten())

else:
    st.warning("Please upload an image file to get a prediction.")

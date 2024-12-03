import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model('/Users/dipeshsharma/Desktop/Hoping Minds/lung_cancer_detection_model.h5')  # Change this to your model's path

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize and convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, (150, 150))  # Resize to match model input
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to create batch
    image /= 255.0  # Normalize the image
    return image

# Title of the web app
st.title("Lung Cancer CT scan Classification")

# Upload file button
uploaded_file = st.file_uploader("Upload your ct scan ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Scan', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make prediction
    if img_array.shape == (1, 150, 150, 3):  # Ensure correct shape
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Output results based on predicted class
        if predicted_class[0] == 0:
            st.write("The scan is normal.")
        elif predicted_class[0] == 1:
            st.write("The scan shows adenocarcinoma.")
        elif predicted_class[0] == 2:
            st.write("The scan shows squamous cell carcinoma.")
        elif predicted_class[0] == 3:
            st.write("The scan shows large cell carcinoma.")
    else:
        st.write("Image preprocessing failed, please check the uploaded file format.")

# Optionally, add more functionality or display results in a formal report format.

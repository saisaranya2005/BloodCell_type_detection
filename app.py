import streamlit as st
from PIL import Image
import torch
import os

def load_model():
    """Load the YOLO model."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def predict(model, image_path):
    """Run predictions on the input image."""
    results = model(image_path)
    return results

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location."""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit UI
st.title("Blood Cell Detection with YOLOv5")

# Load YOLO model
st.sidebar.header("Model Loading")
with st.spinner("Loading YOLOv5 model..."):
    model = load_model()
st.sidebar.success("Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Upload a Blood Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Save the uploaded image
    image_path = save_uploaded_file(uploaded_file)

    # Predict and display results
    st.write("## Predictions")
    with st.spinner("Running YOLOv5 on the uploaded image..."):
        results = predict(model, image_path)

    # Render results
    results.render()  # Save annotated images to results.imgs
    annotated_image = Image.fromarray(results.imgs[0])
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Optional: Display raw prediction data
    if st.checkbox("Show Prediction Details"):
        st.json(results.pandas().xyxy[0].to_dict())

st.sidebar.info("Developed with Streamlit and YOLOv5")

import streamlit as st
from PIL import Image
import torch
import os
from ultralytics import YOLO
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Recycle Detection App", layout="centered")

# Title
st.title("♻️ Recycle Object Detection App")
st.write("Upload an image to detect types of recyclable waste using YOLOv8.")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "yolov8s.pt"
    model = YOLO(model_path)
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Detect button
if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Run detection
    with st.spinner("Detecting..."):
        results = model(temp_path)[0]

    # Draw results
    result_image = results.plot()  # returns numpy array with bounding boxes
    st.image(result_image, caption="Detection Results", use_column_width=True)

    # Display detected classes
    class_names = model.names
    detected_labels = [class_names[int(cls)] for cls in results.boxes.cls]
    if detected_labels:
        st.success("### Detected Items:")
        for label in set(detected_labels):
            count = detected_labels.count(label)
            st.markdown(f"- **{label.capitalize()}** × {count}")
    else:
        st.warning("No recyclable objects detected.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

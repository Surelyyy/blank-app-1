import streamlit as st
from PIL import Image, ImageDraw
import tempfile
from inference_sdk import InferenceHTTPClient

# Roboflow model setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="221fUCs3bTfBgfyCgZ2Z"
)

MODEL_ID = "recyclable-items/3"

# Streamlit UI setup
st.set_page_config(page_title="Recycle Detection App", layout="centered")
st.title("♻️ Recycle Object Detection App")
st.write("Upload an image to detect types of recyclable waste using Roboflow's API.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process image if uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Send to Roboflow for inference
    with st.spinner("Detecting..."):
        result = CLIENT.infer(temp_path, model_id=MODEL_ID)

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    detected_labels = []
    for pred in result['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        class_name = pred['class']
        confidence = pred['confidence']

        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline="lime", width=3)
        draw.text((left, top - 10), f"{class_name} ({confidence:.2f})", fill="lime")
        detected_labels.append(class_name)

    # Display result image
    st.image(image, caption="Detection Results", use_column_width=True)

    # Display detected items
    if detected_labels:
        st.success("### Detected Items:")
        for label in set(detected_labels):
            count = detected_labels.count(label)
            st.markdown(f"- **{label.capitalize()}** × {count}")
    else:
        st.warning("

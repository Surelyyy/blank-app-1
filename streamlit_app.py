import streamlit as st
from PIL import Image, ImageDraw
import tempfile
import requests

# Roboflow model info
MODEL_ID = "recyclable-items/3"
API_KEY = "221fUCs3bTfBgfyCgZ2Z"

# Roboflow API request
def roboflow_infer(image_path, model_id, api_key):
    url = f"https://infer.roboflow.com/{model_id}?api_key={api_key}"
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    return response.json()

# Streamlit UI
st.set_page_config(page_title="Recycle Detection App", layout="centered")
st.title("♻️ Recycle Object Detection App")
st.write("Upload an image to detect types of recyclable waste using Roboflow.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Run detection
    with st.spinner("Detecting..."):
        result = roboflow_infer(temp_path, MODEL_ID, API_KEY)

    # Draw boxes
    draw = ImageDraw.Draw(image)
    detected_labels = []
    for pred in result["predictions"]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2
        label = pred["class"]
        conf = pred["confidence"]
        draw.rectangle([left, top, right, bottom], outline="lime", width=3)
        draw.text((left, top - 10), f"{label} ({conf:.2f})", fill="lime")
        detected_labels.append(label)

    st.image(image, caption="Detection Results", use_column_width=True)

    if detected_labels:
        st.success("### Detected Items:")
        for label in set(detected_labels):
            st.markdown(f"- **{label.capitalize()}** × {detected_labels.count(label)}")
    else:
        st.warning("No recyclable objects detected.")

st.markdown("---")
st.markdown("Made with ❤️ using Roboflow and Streamlit")

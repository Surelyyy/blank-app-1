import streamlit as st
from PIL import Image, ImageDraw
import requests
import tempfile
import base64
import json

# Constants
ROBOFLOW_API_KEY = "221fUCs3bTfBgfyCgZ2Z"
MODEL_ID = "recyclable-items/3"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}?api_key={ROBOFLOW_API_KEY}"

# Streamlit UI
st.set_page_config(page_title="Recycle Detection App", layout="centered")
st.title("♻️ Recycle Object Detection App")
st.write("Upload an image to detect types of recyclable waste using Roboflow API.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    # Read and encode image
    with open(temp_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Send to Roboflow
    with st.spinner("Detecting..."):
        response = requests.post(
            API_URL,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

    if response.status_code == 200:
        result = response.json()
        st.subheader("📦 Detection Results")

        if "predictions" in result and result["predictions"]:
            draw = ImageDraw.Draw(image)
            labels = []

            for pred in result["predictions"]:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2
                label = pred["class"]
                confidence = pred["confidence"]

                draw.rectangle([left, top, right, bottom], outline="lime", width=2)
                draw.text((left, top - 10), f"{label} ({confidence:.2f})", fill="lime")
                labels.append(label)

            st.image(image, caption="Detection Results", use_column_width=True)

            st.success("### Detected Items:")
            for label in set(labels):
                st.markdown(f"- **{label.capitalize()}** × {labels.count(label)}")
        else:
            st.warning("No recyclable items detected.")
    else:
        st.error("Failed to get predictions. Please check your API key or model ID.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Roboflow and Streamlit")

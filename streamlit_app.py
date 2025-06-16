import streamlit as st
from PIL import Image, ImageDraw
import tempfile
import requests

# Roboflow model info
MODEL_ID = "recyclable-items/3"
API_KEY = "221fUCs3bTfBgfyCgZ2Z"

def roboflow_infer(image_path, model_id, api_key):
    url = f"https://infer.roboflow.com/{model_id}?api_key={api_key}&confidence=0.4"
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    try:
        return response.json()
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}"}

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

    # Debug: Show raw API response
    st.subheader("🔍 Raw Roboflow API Response")
    st.json(result)

    # Safe check for predictions
    if "predictions" in result:
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
    else:
        st.error("❌ Roboflow API did not return predictions. Check credentials, model ID, or API status.")

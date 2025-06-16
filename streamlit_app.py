import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
import tempfile

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="221fUCs3bTfBgfyCgZ2Z"
)

# Streamlit UI
st.set_page_config(page_title="Recycle Detection App", layout="centered")
st.title("♻️ Recycle Object Detection App")
st.write("Upload an image to detect types of recyclable waste using Roboflow.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    # Inference
    with st.spinner("Detecting..."):
        result = CLIENT.infer(temp_path, model_id="recyclable-items/3")

    # Show raw response
    st.subheader("🔍 Raw Roboflow API Response")
    st.json(result)

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

            draw.rectangle([left, top, right, bottom], outline="lime", width=2)
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
        st.error("❌ No predictions returned. Please verify model ID and API key.")

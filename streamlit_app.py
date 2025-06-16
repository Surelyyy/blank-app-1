import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- CONFIG ---
MODEL_WEIGHTS = "best_recycle_model.pt"  # path to your YOLOv8 .pt weights
CLASS_NAMES = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']

# --- LOAD MODEL ---
@st.cache_resource
def load_model(weights_path):
    model = YOLO(weights_path)
    # replace model.names if you want to override any labels
    model.names = {i: name for i, name in enumerate(CLASS_NAMES)}
    return model

model = load_model(MODEL_WEIGHTS)

# --- STREAMLIT UI ---
st.title("♻️ Recycle Object Detection")
st.markdown(
    """
Upload an image of waste material, and the model will identify and count items in these categories:
glass, medical, metal, organic, paper, plastic, sharp-object.
"""
)

uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run inference
    with st.spinner("Running YOLOv8..."):
        results = model.predict(source=img_array, imgsz=640, conf=0.25, device=0)[0]

    # Annotate and display image
    annotated = results.plot()  # returns np.ndarray with boxes+labels drawn
    st.image(annotated, caption="Detected objects", use_column_width=True)

    # Summarize detections
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        name = model.names[cls_id]
        counts[name] = counts.get(name, 0) + 1

    if counts:
        st.subheader("Detections Summary")
        for cls in CLASS_NAMES:
            cnt = counts.get(cls, 0)
            st.write(f"- **{cls.capitalize():12s}** : {cnt}")
    else:
        st.info("No objects detected. Try a different image or lower the confidence threshold.")

    # Optional: show raw confidence scores
    if st.checkbox("Show raw detections"):
        st.table([
            {
                "class": model.names[int(box.cls.cpu().numpy())],
                "confidence": float(box.conf.cpu().numpy().round(3)),
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3]),
            }
            for box in results.boxes
        ])

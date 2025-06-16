import os

# ──────────────────────────────────────────────────────
# Disable Streamlit’s file‑watcher and torch.classes introspection
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
torch.classes.__path__ = []
# ──────────────────────────────────────────────────────

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# — your 7 classes —
CLASS_NAMES = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
MODEL_WEIGHTS = "yolov8s.pt"  # make sure this file lives at the repo root

@st.cache_resource
def load_model():
    model = YOLO(MODEL_WEIGHTS)
    model.names = {i: name for i, name in enumerate(CLASS_NAMES)}
    return model

model = load_model()

st.title("♻️ Recycle Object Detection")
st.write("Upload an image of waste, and let YOLOv8 sort it into:\n\n" +
         ", ".join(CLASS_NAMES))

upload = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
if upload:
    img = Image.open(upload).convert("RGB")
    arr = np.array(img)

    with st.spinner("Detecting…"):
        results = model.predict(source=arr, imgsz=640, conf=0.25)[0]

    # draw boxes
    annotated = results.plot()
    st.image(annotated, caption="Detections", use_column_width=True)

    # summary
    counts = {}
    for b in results.boxes:
        cid = int(b.cls.cpu().numpy())
        counts[model.names[cid]] = counts.get(model.names[cid], 0) + 1

    if counts:
        st.subheader("Summary")
        for cls in CLASS_NAMES:
            st.write(f"- **{cls}**: {counts.get(cls, 0)}")
    else:
        st.info("No objects detected. Try a different image or lower the confidence.")

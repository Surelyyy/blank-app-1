# ♻️ Recycle Object Detection

This Streamlit app lets users upload an image of waste and uses a YOLOv8 model
to classify and count items in 7 recycle‐related categories.

## Quickstart

1. **Clone** this repo.
2. **Place** your `best_recycle_model.pt` in the root.
3. **Build & Run** with Docker:

   ```bash
   docker build -t recycle‑detector .
   docker run -p 8501:8501 recycle‑detector

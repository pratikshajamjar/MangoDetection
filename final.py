import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import time
import pandas as pd  # <- Add pandas for table display

# Set Streamlit page config
st.set_page_config(page_title="Mango Ripeness Detection", layout="wide")

# Load YOLO model
model_path = "bestest.pt"
if not os.path.exists(model_path):
    st.error("Model file not found! Please place 'bestest.pt' in the same directory.")
    st.stop()

with st.spinner("Loading YOLO model..."):
    model = YOLO(model_path)
    class_list = model.names

# Color definitions for bounding boxes
color_map = {
    "ripe": (0, 255, 255),
    "unripe": (0, 255, 0),
    "overripe": (42, 42, 165),
    "default": (255, 0, 0)
}

# App title
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Fruit Ripeness Detection</h1>", unsafe_allow_html=True)

# Detection + drawing function
def detect_and_draw(model, image_np, class_list, scale=0.5):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = model.predict(image_bgr)

    object_counts = {}
    total_objects = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls = int(box.cls[0].item())

            class_name = class_list.get(cls, f"Unknown-{cls}")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            total_objects += 1

            label = f"{class_name} ({conf:.2f})"
            color = color_map.get(class_name.lower(), color_map["default"])

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(image_bgr, f"Total Objects: {total_objects}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize image
    new_width = int(image_bgr.shape[1] * scale)
    new_height = int(image_bgr.shape[0] * scale)
    resized_bgr = cv2.resize(image_bgr, (new_width, new_height))

    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, object_counts, total_objects

# Upload image
uploaded_file = st.file_uploader("Upload a fruit image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    with st.spinner("Detecting ripeness..."):
        time.sleep(1)
        output_image, object_counts, total_objects = detect_and_draw(model, image_np, class_list)

    # Side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Original Image")
        st.image(image_np, use_container_width=True)

    with col2:
        st.markdown("Detected Output")
        st.image(output_image, use_container_width=True)

    # Detection summary
    st.markdown("### Detection Summary")
    if total_objects == 0:
        st.warning("No fruits detected in the image.")
    else:
        # Create a DataFrame for tabular display
        df_summary = pd.DataFrame(list(object_counts.items()), columns=["Ripeness Stage", "Count"], index=np.arange(1, len(object_counts) + 1))
        st.table(df_summary)

else:
    st.info("Please upload a fruit image to begin.")

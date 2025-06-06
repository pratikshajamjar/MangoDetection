from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="C:/Users/HP/OneDrive/Desktop/fruit/Fruits-by-YOLO-1/data.yaml",
    epochs=1,
    imgsz=640,
    plots=True
)

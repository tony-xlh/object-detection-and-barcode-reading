from ultralytics import YOLO
# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
success = model.export(format='onnx')
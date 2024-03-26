from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
# from ndarray
im2 = cv2.imread("IMG20240326163255.jpg")
results = model.predict(source=im2)  # save predictions as labels
print(results)
